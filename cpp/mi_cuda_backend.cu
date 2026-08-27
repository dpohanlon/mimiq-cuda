#include "mi_cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
                "cudaMalloc");
        }
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

   private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

struct DeviceRect {
    int16_t x0;
    int16_t x1;
    int16_t y0;
    int16_t y1;
    int16_t depth;
};

__device__ __forceinline__ unsigned int prefix_at(
    const unsigned int* prefix, int stride, int x, int y) {
    if (x < 0 || y < 0) return 0;
    return prefix[x * stride + y];
}

__device__ __forceinline__ unsigned int rectangle_sum(
    const unsigned int* prefix,
    int stride,
    int x0,
    int x1,
    int y0,
    int y1) {
    if (x0 > x1 || y0 > y1) return 0;
    return prefix_at(prefix, stride, x1, y1) -
           prefix_at(prefix, stride, x0 - 1, y1) -
           prefix_at(prefix, stride, x1, y0 - 1) +
           prefix_at(prefix, stride, x0 - 1, y0 - 1);
}

__device__ __forceinline__ void pair_from_linear(
    unsigned long long pair,
    int features,
    int& i,
    int& j) {
    // Consumer Blackwell has deliberately limited FP64 throughput.  A float
    // square root is sufficient for the initial triangular-row estimate;
    // the integer corrections below make the mapping exact even at a
    // floating-point row boundary.
    const unsigned long long discriminant =
        4ULL * static_cast<unsigned long long>(features) *
            static_cast<unsigned long long>(features - 1) -
        7ULL - 8ULL * pair;
    const float root = sqrtf(static_cast<float>(discriminant));
    i = features - 2 -
        static_cast<int>(floorf(root * 0.5F - 0.5F));
    i = max(0, min(i, features - 2));
    unsigned long long first =
        static_cast<unsigned long long>(i) *
        static_cast<unsigned long long>(2 * features - i - 1) / 2ULL;
    while (i > 0 && pair < first) {
        --i;
        first =
            static_cast<unsigned long long>(i) *
            static_cast<unsigned long long>(2 * features - i - 1) / 2ULL;
    }
    while (i + 1 < features - 1) {
        const unsigned long long next =
            static_cast<unsigned long long>(i + 1) *
            static_cast<unsigned long long>(2 * features - i - 2) / 2ULL;
        if (pair < next) break;
        ++i;
        first = next;
    }
    j = i + 1 + static_cast<int>(pair - first);
}

__global__ void histogram_mi_kernel(
    const uint8_t* __restrict__ ranks,
    const int* __restrict__ n_bins,
    const double* __restrict__ boundaries,
    int samples,
    int features,
    int rank_stride,
    int min_pop,
    unsigned long long pairs,
    double* __restrict__ mi_out,
    double* __restrict__ chi2_out) {

    const unsigned long long pair =
        static_cast<unsigned long long>(blockIdx.x);
    if (pair >= pairs) return;

    const int grid_stride = rank_stride;

    int feature_x, feature_y;
    pair_from_linear(pair, features, feature_x, feature_y);
    const int nx = n_bins[feature_x];
    const int ny = n_bins[feature_y];

    extern __shared__ unsigned char shared_raw[];
    auto* prefix = reinterpret_cast<unsigned int*>(shared_raw);
    const int grid_cells = grid_stride * grid_stride;
    auto* stack = reinterpret_cast<DeviceRect*>(
        prefix + grid_cells);
    const int stack_capacity = 2 * rank_stride + 8;
    auto* slices = reinterpret_cast<unsigned int*>(
        stack + stack_capacity);

    for (int cell = threadIdx.x; cell < grid_cells; cell += blockDim.x) {
        prefix[cell] = 0;
    }
    __syncthreads();

    const uint8_t* x =
        ranks + static_cast<size_t>(feature_x) * samples;
    const uint8_t* y =
        ranks + static_cast<size_t>(feature_y) * samples;

    constexpr unsigned int full_mask = 0xffffffffU;
    const int lane = threadIdx.x & 31;
    // The zero/zero state is often overwhelmingly common in sparse counts.
    // Per-thread aggregation avoids that hotspot; native Blackwell shared
    // atomics are faster than warp key matching for the remaining bins.
    unsigned int low_low = 0U;
    for (int row = threadIdx.x; row < samples; row += blockDim.x) {
        const unsigned int rx = x[row];
        const unsigned int ry = y[row];
        if (rx == 0U && ry == 0U) {
            ++low_low;
        } else {
            atomicAdd(prefix + rx * grid_stride + ry, 1U);
        }
    }
    if (low_low != 0U) atomicAdd(prefix, low_low);
    __syncthreads();

    // Inclusive 2-D prefix sum: independent rows, then independent columns.
    // The original implementation did all 4096 cells in one thread.
    for (int ix = threadIdx.x; ix < nx; ix += blockDim.x) {
        unsigned int running = 0;
        for (int iy = 0; iy < ny; ++iy) {
            running += prefix[ix * grid_stride + iy];
            prefix[ix * grid_stride + iy] = running;
        }
    }
    __syncthreads();
    for (int iy = threadIdx.x; iy < ny; iy += blockDim.x) {
        unsigned int running = 0;
        for (int ix = 0; ix < nx; ++ix) {
            running += prefix[ix * grid_stride + iy];
            prefix[ix * grid_stride + iy] = running;
        }
    }
    __syncthreads();

    const double* bx =
        boundaries +
        static_cast<size_t>(feature_x) * (rank_stride + 1);
    const double* by =
        boundaries +
        static_cast<size_t>(feature_y) * (rank_stride + 1);

    // Warp 0 cooperatively evaluates rectangle slices and weighted-median
    // cuts.  Lane 0 retains the exact DFS order and FP64 accumulation order.
    if (threadIdx.x >= 32) return;

    int stack_size = 0;
    double mi = 0.0;
    double chi2 = 0.0;
    if (lane == 0) {
        stack[stack_size++] = DeviceRect{
            0, static_cast<int16_t>(nx - 1),
            0, static_cast<int16_t>(ny - 1), 0};
    }

    while (true) {
        DeviceRect r{};
        int has_work = 0;
        if (lane == 0 && stack_size > 0) {
            r = stack[--stack_size];
            has_work = 1;
        }
        has_work = __shfl_sync(full_mask, has_work, 0);
        if (!has_work) break;
        r.x0 = static_cast<int16_t>(
            __shfl_sync(full_mask, static_cast<int>(r.x0), 0));
        r.x1 = static_cast<int16_t>(
            __shfl_sync(full_mask, static_cast<int>(r.x1), 0));
        r.y0 = static_cast<int16_t>(
            __shfl_sync(full_mask, static_cast<int>(r.y0), 0));
        r.y1 = static_cast<int16_t>(
            __shfl_sync(full_mask, static_cast<int>(r.y1), 0));
        r.depth = static_cast<int16_t>(
            __shfl_sync(full_mask, static_cast<int>(r.depth), 0));

        unsigned int mass = lane == 0
            ? rectangle_sum(
                  prefix, grid_stride, r.x0, r.x1, r.y0, r.y1)
            : 0U;
        mass = __shfl_sync(full_mask, mass, 0);
        if (mass == 0) continue;

        int axis = r.depth & 1;
        int lo = 0;
        int hi = -1;
        int splittable = 0;
        for (int attempt = 0; attempt < 2 && !splittable; ++attempt) {
            const int begin = axis == 0 ? r.x0 : r.y0;
            const int end = axis == 0 ? r.x1 : r.y1;
            int local_lo = 0x7fffffff;
            int local_hi = -1;
            for (int coord = begin + lane;
                 coord <= end;
                 coord += 32) {
                const unsigned int slice = axis == 0
                    ? rectangle_sum(
                          prefix, grid_stride, coord, coord, r.y0, r.y1)
                    : rectangle_sum(
                          prefix, grid_stride, r.x0, r.x1, coord, coord);
                if (slice != 0U) {
                    local_lo = min(local_lo, coord);
                    local_hi = max(local_hi, coord);
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_lo = min(
                    local_lo,
                    __shfl_down_sync(full_mask, local_lo, offset));
                local_hi = max(
                    local_hi,
                    __shfl_down_sync(full_mask, local_hi, offset));
            }
            lo = __shfl_sync(full_mask, local_lo, 0);
            hi = __shfl_sync(full_mask, local_hi, 0);
            splittable = lo < hi;
            if (!splittable) axis = 1 - axis;
        }

        if (mass <= static_cast<unsigned int>(min_pop) || !splittable) {
            if (lane == 0) {
                const double p =
                    static_cast<double>(mass) /
                    static_cast<double>(samples);
                double q =
                    (bx[r.x1 + 1] - bx[r.x0]) *
                    (by[r.y1 + 1] - by[r.y0]);
                q = fmax(q, 1e-300);
                // One FP64 logarithm instead of log(p) - log(q).  p/q is
                // bounded by the normal double range after q is clamped.
                mi += p * log(p / q);
                const double expected = static_cast<double>(samples) * q;
                if (expected > 0.0) {
                    const double diff =
                        static_cast<double>(mass) - expected;
                    chi2 += diff * diff / expected;
                }
            }
            continue;
        }

        const unsigned int target = (mass + 1U) / 2U;
        for (int coord = lo + lane; coord <= hi; coord += 32) {
            const unsigned int slice = axis == 0
                ? rectangle_sum(
                      prefix, grid_stride, coord, coord, r.y0, r.y1)
                : rectangle_sum(
                      prefix, grid_stride, r.x0, r.x1, coord, coord);
            slices[coord] = slice;
        }
        __syncwarp();

        unsigned int carry = 0U;
        int cut = lo + 1;
        int found = 0;
        for (int chunk = lo; chunk <= hi && !found; chunk += 32) {
            const int coord = chunk + lane;
            const unsigned int slice =
                coord <= hi ? slices[coord] : 0U;
            unsigned int cumulative = slice;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned int previous =
                    __shfl_up_sync(full_mask, cumulative, offset);
                if (lane >= offset) cumulative += previous;
            }
            const unsigned int candidate_mask = __ballot_sync(
                full_mask,
                coord <= hi && coord > lo && slice != 0U &&
                    carry + cumulative >= target);
            if (candidate_mask != 0U) {
                cut = chunk + __ffs(static_cast<int>(candidate_mask)) - 1;
                found = 1;
            }
            carry += __shfl_sync(full_mask, cumulative, 31);
        }
        if (!found) {
            int local_cut = 0x7fffffff;
            for (int coord = lo + 1 + lane;
                 coord <= hi;
                 coord += 32) {
                if (slices[coord] != 0U) {
                    local_cut = min(local_cut, coord);
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_cut = min(
                    local_cut,
                    __shfl_down_sync(full_mask, local_cut, offset));
            }
            cut = __shfl_sync(full_mask, local_cut, 0);
        }

        if (lane == 0) {
            DeviceRect left = r;
            DeviceRect right = r;
            left.depth = right.depth =
                static_cast<int16_t>(r.depth + 1);
            if (axis == 0) {
                left.x1 = static_cast<int16_t>(cut - 1);
                right.x0 = static_cast<int16_t>(cut);
            } else {
                left.y1 = static_cast<int16_t>(cut - 1);
                right.y0 = static_cast<int16_t>(cut);
            }
            // Preserve the reference DFS order exactly.
            stack[stack_size++] = right;
            stack[stack_size++] = left;
        }
        __syncwarp();
    }

    if (lane == 0) {
        mi_out[pair] = mi;
        chi2_out[pair] = chi2;
    }
}

}  // namespace

bool mi_cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string mi_cuda_device_name() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    return properties.name;
}

CudaMIOutput run_cuda_histogram_mi(
    const CompressedMIData<uint8_t>& data,
    int min_pop) {

    if (!mi_cuda_available()) {
        throw std::runtime_error("no CUDA-capable device is available");
    }
    if (data.rank_stride <= 0 || data.rank_stride > 64) {
        throw std::invalid_argument(
            "CUDA histogram backend requires between 1 and 64 bins");
    }

    const unsigned long long pair_count =
        static_cast<unsigned long long>(data.n_features) *
        static_cast<unsigned long long>(data.n_features - 1) / 2ULL;
    if (pair_count >
        static_cast<unsigned long long>(
            std::numeric_limits<unsigned int>::max())) {
        throw std::invalid_argument("too many feature pairs for one CUDA launch");
    }

    DeviceBuffer<uint8_t> d_ranks(data.ranks.size());
    DeviceBuffer<int> d_bins(data.n_bins.size());
    DeviceBuffer<double> d_boundaries(data.cdf_boundaries.size());
    DeviceBuffer<double> d_mi(static_cast<size_t>(pair_count));
    DeviceBuffer<double> d_chi2(static_cast<size_t>(pair_count));

    check_cuda(
        cudaMemcpy(
            d_ranks.get(), data.ranks.data(),
            data.ranks.size() * sizeof(uint8_t), cudaMemcpyHostToDevice),
        "copy ranks to CUDA device");
    check_cuda(
        cudaMemcpy(
            d_bins.get(), data.n_bins.data(),
            data.n_bins.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy bin counts to CUDA device");
    check_cuda(
        cudaMemcpy(
            d_boundaries.get(), data.cdf_boundaries.data(),
            data.cdf_boundaries.size() * sizeof(double),
            cudaMemcpyHostToDevice),
        "copy CDF boundaries to CUDA device");

    constexpr int threads = 256;
    const size_t stack_capacity =
        static_cast<size_t>(2 * data.rank_stride + 8);
    const size_t shared_bytes =
        static_cast<size_t>(data.rank_stride) * data.rank_stride *
            sizeof(unsigned int) +
        stack_capacity * sizeof(DeviceRect) +
        static_cast<size_t>(data.rank_stride) * sizeof(unsigned int);

    histogram_mi_kernel<<<
        static_cast<unsigned int>(pair_count), threads, shared_bytes>>>(
        d_ranks.get(), d_bins.get(), d_boundaries.get(),
        data.n_samples, data.n_features, data.rank_stride, min_pop,
        pair_count, d_mi.get(), d_chi2.get());
    check_cuda(cudaGetLastError(), "launch histogram MI CUDA kernel");
    check_cuda(cudaDeviceSynchronize(), "execute histogram MI CUDA kernel");

    CudaMIOutput output;
    output.mi.resize(static_cast<size_t>(pair_count));
    output.chi2.resize(static_cast<size_t>(pair_count));
    check_cuda(
        cudaMemcpy(
            output.mi.data(), d_mi.get(),
            output.mi.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy MI from CUDA device");
    check_cuda(
        cudaMemcpy(
            output.chi2.data(), d_chi2.get(),
            output.chi2.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy chi-square from CUDA device");
    return output;
}
