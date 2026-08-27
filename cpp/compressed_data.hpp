#pragma once

#include <vector>

template <typename RankT>
struct CompressedMIData {
    int n_samples = 0;
    int n_features = 0;
    int rank_stride = 0;
    std::vector<RankT> ranks;             // gene-major: [feature][sample]
    std::vector<int> n_bins;
    std::vector<double> cdf_boundaries;   // [feature][0..rank_stride]
};
