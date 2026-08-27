#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "compressed_data.hpp"

struct CudaMIOutput {
    std::vector<double> mi;
    std::vector<double> chi2;
};

bool mi_cuda_available();
std::string mi_cuda_device_name();

CudaMIOutput run_cuda_histogram_mi(
    const CompressedMIData<uint8_t>& data,
    int min_pop);
