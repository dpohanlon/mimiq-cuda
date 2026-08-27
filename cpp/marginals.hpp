#pragma once

#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/beta.hpp>

inline double normal_cdf_discrete(int value, double mean, double std_dev) {
    const boost::math::normal distribution(mean, std_dev);
    return boost::math::cdf(distribution, value + 0.5);
}

inline double nb2_cdf_single(
    int value, double mean, double concentration) {
    const double probability = concentration / (mean + concentration);
    return boost::math::ibeta(
        concentration, static_cast<double>(value) + 1.0, probability);
}

inline double zinb2_cdf_single(
    int value,
    double mean,
    double concentration,
    double zero_probability) {
    return zero_probability +
        (1.0 - zero_probability) *
            nb2_cdf_single(value, mean, concentration);
}
