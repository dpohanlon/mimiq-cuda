#!/usr/bin/env python3
"""Reproducible CPU/GPU timing and equivalence benchmark."""

import argparse
import statistics
import time

import numpy as np

import mimiq_cuda


def timed(function, repeats):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - start)
    return result, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=50_000)
    parser.add_argument("--genes", type=int, default=1_000)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    concentration = np.exp(
        rng.uniform(np.log(1.5), np.log(30.0), args.genes)
    )
    mean = np.exp(rng.uniform(np.log(0.2), np.log(80.0), args.genes))
    probability = concentration / (concentration + mean)
    counts = rng.negative_binomial(
        concentration,
        probability,
        size=(args.cells, args.genes),
    ).astype(np.int32)

    gpu_fn = lambda: mimiq_cuda.mi_negative_binomial_gpu(
        counts, mean, concentration, max_bins=args.bins
    )
    # Warm up context creation and PTX/cubin loading outside measurements.
    mimiq_cuda.mi_negative_binomial_gpu(
        counts[: min(args.cells, 2048), : min(args.genes, 64)],
        mean[: min(args.genes, 64)],
        concentration[: min(args.genes, 64)],
        max_bins=args.bins,
    )
    gpu, gpu_times = timed(gpu_fn, args.repeats)
    print(
        f"GPU median={statistics.median(gpu_times):.6f}s "
        f"runs={gpu_times}"
    )

    if args.cpu:
        cpu, cpu_times = timed(
            lambda: mimiq_cuda.mi_negative_binomial_cpu(
                counts, mean, concentration, max_bins=args.bins
            ),
            args.repeats,
        )
        errors = [
            float(np.max(np.abs(actual - expected)))
            for actual, expected in zip(gpu, cpu)
        ]
        print(
            f"CPU median={statistics.median(cpu_times):.6f}s "
            f"runs={cpu_times}"
        )
        print(f"max_abs_errors={errors}")


if __name__ == "__main__":
    main()
