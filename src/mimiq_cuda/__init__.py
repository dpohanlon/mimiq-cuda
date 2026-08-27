"""Fast histogram-native pairwise mutual information on CPU and CUDA."""

from ._core import (
    cuda_available,
    cuda_device_name,
    mi_negative_binomial_cpu,
    mi_negative_binomial_gpu,
    mi_negative_binomial_zi_cpu,
    mi_negative_binomial_zi_gpu,
    mi_normal_cpu,
    mi_normal_gpu,
)

__version__ = "0.1.0"

# Explicit compatibility aliases for the accelerated histogram API developed
# in the original mimiq repository.
mi_normal_hist = mi_normal_cpu
mi_negative_binomial_hist = mi_negative_binomial_cpu
mi_negative_binomial_zi_hist = mi_negative_binomial_zi_cpu
mi_normal_hist_gpu = mi_normal_gpu
mi_negative_binomial_hist_gpu = mi_negative_binomial_gpu
mi_negative_binomial_zi_hist_gpu = mi_negative_binomial_zi_gpu

__all__ = [
    "__version__",
    "cuda_available",
    "cuda_device_name",
    "mi_normal_cpu",
    "mi_normal_gpu",
    "mi_negative_binomial_cpu",
    "mi_negative_binomial_gpu",
    "mi_negative_binomial_zi_cpu",
    "mi_negative_binomial_zi_gpu",
    "mi_normal_hist",
    "mi_normal_hist_gpu",
    "mi_negative_binomial_hist",
    "mi_negative_binomial_hist_gpu",
    "mi_negative_binomial_zi_hist",
    "mi_negative_binomial_zi_hist_gpu",
]
