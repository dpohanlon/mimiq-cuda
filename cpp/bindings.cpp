#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "mi_cuda_backend.hpp"
#include "histogram_mi.hpp"

namespace py = pybind11;

namespace {

using IntMatrixArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleVectorArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

Eigen::MatrixXi as_integer_matrix(
    const IntMatrixArray& array, const char* name) {
    if (array.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be two-dimensional");
    }
    const auto input = array.unchecked<2>();
    if (input.shape(0) == 0 || input.shape(1) < 2) {
        throw py::value_error(
            std::string(name) +
            " must contain at least one row and two feature columns");
    }
    Eigen::MatrixXi output(input.shape(0), input.shape(1));
    for (py::ssize_t row = 0; row < input.shape(0); ++row) {
        for (py::ssize_t column = 0; column < input.shape(1); ++column) {
            const int value = input(row, column);
            if (value < 0) {
                throw py::value_error(
                    std::string(name) + " must contain non-negative counts");
            }
            output(row, column) = value;
        }
    }
    return output;
}

Eigen::VectorXd as_double_vector(
    const DoubleVectorArray& array,
    int expected,
    const char* name,
    bool strictly_positive) {
    if (array.ndim() != 1 || array.shape(0) != expected) {
        throw py::value_error(
            std::string(name) + " must match the number of features");
    }
    const auto input = array.unchecked<1>();
    Eigen::VectorXd output(expected);
    for (int index = 0; index < expected; ++index) {
        const double value = input(index);
        if (!std::isfinite(value) ||
            (strictly_positive && !(value > 0.0))) {
            throw py::value_error(
                std::string(name) + " contains an invalid value");
        }
        output(index) = value;
    }
    return output;
}

void check_min_pop(int min_pop) {
    if (min_pop <= 0) {
        throw py::value_error("min_pop must be positive");
    }
}

void check_cpu_bins(int max_bins) {
    if (max_bins < 0 || max_bins > 65535) {
        throw py::value_error(
            "max_bins must be zero or between 1 and 65535");
    }
}

void check_gpu_bins(int max_bins) {
    if (max_bins < 1 || max_bins > 64) {
        throw py::value_error("GPU max_bins must be between 1 and 64");
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> unpack_cuda(
    const CudaMIOutput& packed, int features) {
    Eigen::MatrixXd mi = Eigen::MatrixXd::Zero(features, features);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(features, features);
    std::size_t pair = 0;
    for (int left = 0; left < features; ++left) {
        for (int right = left + 1; right < features; ++right, ++pair) {
            mi(left, right) = packed.mi[pair];
            chi2(left, right) = packed.chi2[pair];
        }
    }
    return {std::move(mi), std::move(chi2)};
}

template <typename CDF>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> gpu_mi(
    const Eigen::MatrixXi& data,
    int min_pop,
    int max_bins,
    const CDF& cdf) {
    const auto compressed =
        compress_mi_features<std::uint8_t>(data, max_bins, cdf);
    return unpack_cuda(
        run_cuda_histogram_mi(compressed, min_pop), data.cols());
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() =
        "Portable C++/CUDA histogram-native pairwise mutual information";

    module.def("cuda_available", &mi_cuda_available);
    module.def("cuda_device_name", &mi_cuda_device_name);

    module.def(
        "mi_normal_cpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", false);
            const Eigen::VectorXd scale_values =
                as_double_vector(
                    std_devs, data.cols(), "std_devs", true);
            check_min_pop(min_pop);
            check_cpu_bins(max_bins);
            py::gil_scoped_release release;
            return mutual_information_normal_histogram(
                data, mean_values, scale_values, min_pop, max_bins);
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("std_devs"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 0);

    module.def(
        "mi_negative_binomial_cpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", true);
            const Eigen::VectorXd concentration_values =
                as_double_vector(
                    concentrations,
                    data.cols(),
                    "concentrations",
                    true);
            check_min_pop(min_pop);
            check_cpu_bins(max_bins);
            py::gil_scoped_release release;
            return mutual_information_nb_histogram(
                data,
                mean_values,
                concentration_values,
                min_pop,
                max_bins);
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 0);

    module.def(
        "mi_negative_binomial_zi_cpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& zero_probabilities,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", true);
            const Eigen::VectorXd concentration_values =
                as_double_vector(
                    concentrations,
                    data.cols(),
                    "concentrations",
                    true);
            const Eigen::VectorXd zero_values =
                as_double_vector(
                    zero_probabilities,
                    data.cols(),
                    "zero_probabilities",
                    false);
            for (int index = 0; index < zero_values.size(); ++index) {
                if (zero_values(index) < 0.0 ||
                    zero_values(index) > 1.0) {
                    throw py::value_error(
                        "zero_probabilities must lie in [0, 1]");
                }
            }
            check_min_pop(min_pop);
            check_cpu_bins(max_bins);
            py::gil_scoped_release release;
            return mutual_information_zinb_histogram(
                data,
                mean_values,
                concentration_values,
                zero_values,
                min_pop,
                max_bins);
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("zero_probabilities"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 0);

    module.def(
        "mi_normal_gpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", false);
            const Eigen::VectorXd scale_values =
                as_double_vector(
                    std_devs, data.cols(), "std_devs", true);
            check_min_pop(min_pop);
            check_gpu_bins(max_bins);
            py::gil_scoped_release release;
            return gpu_mi(
                data,
                min_pop,
                max_bins,
                [&](int feature, int value) {
                    return value < 0
                        ? 0.0
                        : normal_cdf_discrete(
                              value,
                              mean_values(feature),
                              scale_values(feature));
                });
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("std_devs"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 64);

    module.def(
        "mi_negative_binomial_gpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", true);
            const Eigen::VectorXd concentration_values =
                as_double_vector(
                    concentrations,
                    data.cols(),
                    "concentrations",
                    true);
            check_min_pop(min_pop);
            check_gpu_bins(max_bins);
            py::gil_scoped_release release;
            return gpu_mi(
                data,
                min_pop,
                max_bins,
                [&](int feature, int value) {
                    return value < 0
                        ? 0.0
                        : nb2_cdf_single(
                              value,
                              mean_values(feature),
                              concentration_values(feature));
                });
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 64);

    module.def(
        "mi_negative_binomial_zi_gpu",
        [](const IntMatrixArray& counts,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& zero_probabilities,
           int min_pop,
           int max_bins) {
            const Eigen::MatrixXi data =
                as_integer_matrix(counts, "counts");
            const Eigen::VectorXd mean_values =
                as_double_vector(means, data.cols(), "means", true);
            const Eigen::VectorXd concentration_values =
                as_double_vector(
                    concentrations,
                    data.cols(),
                    "concentrations",
                    true);
            const Eigen::VectorXd zero_values =
                as_double_vector(
                    zero_probabilities,
                    data.cols(),
                    "zero_probabilities",
                    false);
            for (int index = 0; index < zero_values.size(); ++index) {
                if (zero_values(index) < 0.0 ||
                    zero_values(index) > 1.0) {
                    throw py::value_error(
                        "zero_probabilities must lie in [0, 1]");
                }
            }
            check_min_pop(min_pop);
            check_gpu_bins(max_bins);
            py::gil_scoped_release release;
            return gpu_mi(
                data,
                min_pop,
                max_bins,
                [&](int feature, int value) {
                    return value < 0
                        ? 0.0
                        : zinb2_cdf_single(
                              value,
                              mean_values(feature),
                              concentration_values(feature),
                              zero_values(feature));
                });
        },
        py::arg("counts"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("zero_probabilities"),
        py::arg("min_pop") = 25,
        py::arg("max_bins") = 64);
}
