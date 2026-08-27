#pragma once

#include <Eigen/Dense>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "marginals.hpp"
#include "compressed_data.hpp"

// A sort-free implementation of MIMIQ's integer estimator.  Each feature is
// order-compressed once, then each pair is represented by a small contingency
// grid.  The adaptive partition is evaluated directly from its integral image.
// This preserves the weighted-median split and CDF-boundary semantics of the
// integer KDTree when max_bins == 0.

using MarginalCDFFunction = std::function<double(int, int)>;

template <typename RankT>
CompressedMIData<RankT> compress_mi_features(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    int max_bins,
    const MarginalCDFFunction& marginal_cdf) {

    const int n_samples = samples.rows();
    const int n_features = samples.cols();
    const int rank_limit =
        static_cast<int>(std::numeric_limits<RankT>::max()) + 1;
    if (max_bins < 0 || max_bins > rank_limit) {
        throw std::invalid_argument("max_bins exceeds the rank representation");
    }

    struct FeatureEncoding {
        std::vector<RankT> ranks;
        std::vector<double> boundaries;
        int bins = 0;
    };
    std::vector<FeatureEncoding> encoded(n_features);

    int observed_max_bins = 0;

    #pragma omp parallel for schedule(static) reduction(max:observed_max_bins)
    for (int feature = 0; feature < n_features; ++feature) {
        int max_value = 0;
        for (int row = 0; row < n_samples; ++row) {
            max_value = std::max(max_value, samples(row, feature));
        }

        // Direct addressing is markedly faster for the small integer counts
        // MIMIQ targets.  Avoid pathological allocation for unusually large
        // count labels by falling back to a hash table.
        const bool use_direct =
            max_value <= 1'000'000 &&
            static_cast<long long>(max_value) <=
                8LL * static_cast<long long>(n_samples) + 4096LL;

        std::vector<int> values;
        std::vector<int> frequencies;
        if (use_direct) {
            std::vector<int> freq(static_cast<size_t>(max_value) + 1, 0);
            for (int row = 0; row < n_samples; ++row) {
                ++freq[static_cast<size_t>(samples(row, feature))];
            }
            for (int value = 0; value <= max_value; ++value) {
                if (freq[static_cast<size_t>(value)] != 0) {
                    values.push_back(value);
                    frequencies.push_back(freq[static_cast<size_t>(value)]);
                }
            }
        } else {
            std::unordered_map<int, int> freq;
            freq.reserve(static_cast<size_t>(n_samples / 2 + 1));
            for (int row = 0; row < n_samples; ++row) {
                ++freq[samples(row, feature)];
            }
            values.reserve(freq.size());
            for (const auto& entry : freq) values.push_back(entry.first);
            std::sort(values.begin(), values.end());
            frequencies.reserve(values.size());
            for (int value : values) frequencies.push_back(freq[value]);
        }

        const int unique_count = static_cast<int>(values.size());
        const bool bounded = max_bins != 0 && unique_count > max_bins;
        int bins = bounded ? max_bins : unique_count;
        if (bins > rank_limit) {
            // Exceptions may not safely cross an OpenMP worker boundary.
            // Mark the feature and report after preprocessing joins.
            encoded[feature].bins = -1;
            continue;
        }

        // Consecutive ordinal values are grouped only when a bounded grid was
        // explicitly requested (the GPU path). Repeatedly merge the lightest
        // adjacent groups so common count states retain their own bins while
        // long, sparse tails are coarsened. This always uses the full bin
        // budget and never changes count ordering.
        std::vector<int> unique_to_bin(unique_count);
        if (!bounded) {
            for (int u = 0; u < unique_count; ++u) unique_to_bin[u] = u;
        } else {
            struct Group {
                long long mass;
                int first;
                int last;
                int previous;
                int next;
                unsigned int version;
                bool active;
            };
            struct MergeCandidate {
                long long mass;
                int left;
                int right;
                unsigned int left_version;
                unsigned int right_version;
                bool operator>(const MergeCandidate& other) const {
                    return mass > other.mass;
                }
            };

            std::vector<Group> groups(unique_count);
            for (int u = 0; u < unique_count; ++u) {
                groups[u] = Group{
                    frequencies[u], u, u, u - 1,
                    u + 1 < unique_count ? u + 1 : -1, 0U, true};
            }

            std::priority_queue<
                MergeCandidate,
                std::vector<MergeCandidate>,
                std::greater<MergeCandidate>>
                candidates;
            const auto enqueue = [&](int left) {
                if (left < 0 || !groups[left].active) return;
                const int right = groups[left].next;
                if (right < 0 || !groups[right].active) return;
                candidates.push(MergeCandidate{
                    groups[left].mass + groups[right].mass,
                    left,
                    right,
                    groups[left].version,
                    groups[right].version});
            };
            for (int u = 0; u + 1 < unique_count; ++u) enqueue(u);

            int active_groups = unique_count;
            while (active_groups > max_bins) {
                MergeCandidate candidate;
                do {
                    if (candidates.empty()) {
                        throw std::runtime_error(
                            "failed to construct bounded count bins");
                    }
                    candidate = candidates.top();
                    candidates.pop();
                } while (
                    !groups[candidate.left].active ||
                    !groups[candidate.right].active ||
                    groups[candidate.left].next != candidate.right ||
                    groups[candidate.left].version !=
                        candidate.left_version ||
                    groups[candidate.right].version !=
                        candidate.right_version);

                Group& left = groups[candidate.left];
                Group& right = groups[candidate.right];
                const int previous = left.previous;
                const int next = right.next;
                left.mass += right.mass;
                left.last = right.last;
                left.next = next;
                ++left.version;
                right.active = false;
                ++right.version;
                if (next >= 0) {
                    groups[next].previous = candidate.left;
                    ++groups[next].version;
                }
                --active_groups;
                enqueue(previous);
                enqueue(candidate.left);
                enqueue(next);
            }

            int group = 0;
            int current = 0;
            while (groups[current].previous >= 0) {
                current = groups[current].previous;
            }
            while (current >= 0) {
                for (int u = groups[current].first;
                     u <= groups[current].last; ++u) {
                    unique_to_bin[u] = group;
                }
                ++group;
                current = groups[current].next;
            }
            bins = group;
        }
        std::vector<int> group_first_value(bins);
        for (int bin = 0, u = 0; bin < bins; ++bin) {
            while (u < unique_count && unique_to_bin[u] < bin) ++u;
            group_first_value[bin] = values[u];
        }

        auto& out = encoded[feature];
        out.bins = bins;
        out.ranks.resize(n_samples);

        if (use_direct) {
            std::vector<RankT> value_to_bin(
                static_cast<size_t>(max_value) + 1, RankT{0});
            for (int u = 0; u < unique_count; ++u) {
                value_to_bin[static_cast<size_t>(values[u])] =
                    static_cast<RankT>(unique_to_bin[u]);
            }
            for (int row = 0; row < n_samples; ++row) {
                out.ranks[row] =
                    value_to_bin[static_cast<size_t>(samples(row, feature))];
            }
        } else {
            std::unordered_map<int, RankT> value_to_bin;
            value_to_bin.reserve(values.size());
            for (int u = 0; u < unique_count; ++u) {
                value_to_bin.emplace(
                    values[u], static_cast<RankT>(unique_to_bin[u]));
            }
            for (int row = 0; row < n_samples; ++row) {
                out.ranks[row] = value_to_bin.at(samples(row, feature));
            }
        }

        // Boundary b is the copula CDF immediately before bin b.  Boundary
        // bins is the CDF at the observed root maximum, matching KDTree bounds.
        out.boundaries.resize(static_cast<size_t>(bins) + 1);
        for (int bin = 0; bin < bins; ++bin) {
            out.boundaries[bin] =
                marginal_cdf(feature, group_first_value[bin] - 1);
        }
        out.boundaries[bins] = marginal_cdf(feature, values.back());
        observed_max_bins = std::max(observed_max_bins, bins);
    }

    CompressedMIData<RankT> result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.rank_stride = observed_max_bins;
    result.ranks.resize(
        static_cast<size_t>(n_features) * static_cast<size_t>(n_samples));
    result.n_bins.resize(n_features);
    result.cdf_boundaries.assign(
        static_cast<size_t>(n_features) *
            static_cast<size_t>(observed_max_bins + 1),
        0.0);

    for (int feature = 0; feature < n_features; ++feature) {
        const auto& src = encoded[feature];
        if (src.bins < 0) {
            throw std::runtime_error(
                "feature has too many distinct values for rank type");
        }
        std::copy(
            src.ranks.begin(), src.ranks.end(),
            result.ranks.begin() +
                static_cast<size_t>(feature) * n_samples);
        result.n_bins[feature] = src.bins;
        std::copy(
            src.boundaries.begin(), src.boundaries.end(),
            result.cdf_boundaries.begin() +
                static_cast<size_t>(feature) * (observed_max_bins + 1));
    }
    return result;
}

namespace histogram_mi_detail {

struct RankRect {
    int x0, x1, y0, y1, depth;
};

inline long long rectangle_sum(
    const std::vector<long long>& prefix,
    int y_stride,
    int x0,
    int x1,
    int y0,
    int y1) {
    if (x0 > x1 || y0 > y1) return 0;
    const auto at = [&](int x, int y) -> long long {
        if (x < 0 || y < 0) return 0;
        return prefix[static_cast<size_t>(x) * y_stride + y];
    };
    return at(x1, y1) - at(x0 - 1, y1) - at(x1, y0 - 1) +
           at(x0 - 1, y0 - 1);
}

template <typename RankT>
std::pair<double, double> evaluate_pair(
    const CompressedMIData<RankT>& data,
    int feature_x,
    int feature_y,
    int min_pop) {

    const int nx = data.n_bins[feature_x];
    const int ny = data.n_bins[feature_y];
    const int n = data.n_samples;
    const RankT* x =
        data.ranks.data() + static_cast<size_t>(feature_x) * n;
    const RankT* y =
        data.ranks.data() + static_cast<size_t>(feature_y) * n;

    std::vector<long long> prefix(
        static_cast<size_t>(nx) * static_cast<size_t>(ny), 0);
    for (int row = 0; row < n; ++row) {
        ++prefix[static_cast<size_t>(x[row]) * ny + y[row]];
    }
    for (int ix = 0; ix < nx; ++ix) {
        long long row_running = 0;
        for (int iy = 0; iy < ny; ++iy) {
            row_running += prefix[static_cast<size_t>(ix) * ny + iy];
            if (ix > 0) {
                prefix[static_cast<size_t>(ix) * ny + iy] =
                    row_running +
                    prefix[static_cast<size_t>(ix - 1) * ny + iy];
            } else {
                prefix[static_cast<size_t>(ix) * ny + iy] = row_running;
            }
        }
    }

    const int boundary_stride = data.rank_stride + 1;
    const double* bx =
        data.cdf_boundaries.data() +
        static_cast<size_t>(feature_x) * boundary_stride;
    const double* by =
        data.cdf_boundaries.data() +
        static_cast<size_t>(feature_y) * boundary_stride;

    std::vector<RankRect> stack;
    stack.reserve(static_cast<size_t>(nx + ny + 4));
    stack.push_back({0, nx - 1, 0, ny - 1, 0});

    double mi = 0.0;
    double chi2 = 0.0;
    const auto rect_sum = [&](const RankRect& r) {
        return rectangle_sum(prefix, ny, r.x0, r.x1, r.y0, r.y1);
    };
    const auto axis_slice = [&](const RankRect& r, int axis, int coord) {
        return axis == 0
            ? rectangle_sum(prefix, ny, coord, coord, r.y0, r.y1)
            : rectangle_sum(prefix, ny, r.x0, r.x1, coord, coord);
    };

    while (!stack.empty()) {
        const RankRect r = stack.back();
        stack.pop_back();
        const long long mass = rect_sum(r);
        if (mass == 0) continue;

        int axis = r.depth & 1;
        int min_coord = 0;
        int max_coord = -1;
        const auto find_support = [&](int candidate_axis, int& lo, int& hi) {
            const int begin = candidate_axis == 0 ? r.x0 : r.y0;
            const int end = candidate_axis == 0 ? r.x1 : r.y1;
            lo = begin;
            while (lo <= end && axis_slice(r, candidate_axis, lo) == 0) ++lo;
            hi = end;
            while (hi >= lo && axis_slice(r, candidate_axis, hi) == 0) --hi;
            return lo < hi;
        };

        bool splittable = find_support(axis, min_coord, max_coord);
        if (!splittable) {
            axis = 1 - axis;
            splittable = find_support(axis, min_coord, max_coord);
        }

        if (mass <= min_pop || !splittable) {
            const double p =
                static_cast<double>(mass) / static_cast<double>(n);
            double q =
                (bx[r.x1 + 1] - bx[r.x0]) *
                (by[r.y1 + 1] - by[r.y0]);
            q = std::max(q, 1e-300);
            mi += p * (std::log(p) - std::log(q));
            const double expected = static_cast<double>(n) * q;
            if (expected > 0.0) {
                const double diff = static_cast<double>(mass) - expected;
                chi2 += diff * diff / expected;
            }
            continue;
        }

        const long long target = (mass + 1) / 2;
        long long cumulative = 0;
        int cut = min_coord + 1;
        bool found = false;
        for (int coord = min_coord; coord <= max_coord; ++coord) {
            const long long block_mass = axis_slice(r, axis, coord);
            if (block_mass == 0) continue;
            cumulative += block_mass;
            if (cumulative >= target && coord > min_coord) {
                cut = coord;
                found = true;
                break;
            }
        }
        if (!found) {
            cut = min_coord + 1;
            while (cut <= max_coord &&
                   axis_slice(r, axis, cut) == 0) {
                ++cut;
            }
        }

        RankRect left = r;
        RankRect right = r;
        left.depth = right.depth = r.depth + 1;
        if (axis == 0) {
            left.x1 = cut - 1;
            right.x0 = cut;
        } else {
            left.y1 = cut - 1;
            right.y0 = cut;
        }
        stack.push_back(right);
        stack.push_back(left);
    }
    return {mi, chi2};
}

}  // namespace histogram_mi_detail

template <typename RankT>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> evaluate_histogram_mi(
    const CompressedMIData<RankT>& data,
    int min_pop) {

    const int features = data.n_features;
    Eigen::MatrixXd mi = Eigen::MatrixXd::Zero(features, features);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(features, features);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < features; ++i) {
        for (int j = i + 1; j < features; ++j) {
            const auto result =
                histogram_mi_detail::evaluate_pair(data, i, j, min_pop);
            mi(i, j) = result.first;
            chi2(i, j) = result.second;
        }
    }
    return {mi, chi2};
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_normal_histogram(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& std_devs,
    int min_pop,
    int max_bins = 0) {

    const auto cdf = [&](int feature, int value) {
        return value < 0
            ? 0.0
            : normal_cdf_discrete(value, means(feature), std_devs(feature));
    };
    auto compressed =
        compress_mi_features<uint16_t>(samples, max_bins, cdf);
    return evaluate_histogram_mi(compressed, min_pop);
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_nb_histogram(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concentrations,
    int min_pop,
    int max_bins = 0) {

    const auto cdf = [&](int feature, int value) {
        return value < 0
            ? 0.0
            : nb2_cdf_single(
                  value, means(feature), concentrations(feature));
    };
    auto compressed =
        compress_mi_features<uint16_t>(samples, max_bins, cdf);
    return evaluate_histogram_mi(compressed, min_pop);
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_zinb_histogram(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concentrations,
    const Eigen::Ref<const Eigen::VectorXd>& alphas,
    int min_pop,
    int max_bins = 0) {

    const auto cdf = [&](int feature, int value) {
        return value < 0
            ? 0.0
            : zinb2_cdf_single(
                  value, means(feature), concentrations(feature),
                  alphas(feature));
    };
    auto compressed =
        compress_mi_features<uint16_t>(samples, max_bins, cdf);
    return evaluate_histogram_mi(compressed, min_pop);
}
