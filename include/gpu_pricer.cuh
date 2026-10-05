#pragma once

// Host-only declarations: this header is included from plain .cpp files built
// by MSVC/gcc, so nothing here may use __global__/__device__.

#include <cmath>
#include <cstdint>

#include "option.hpp"

struct GpuTiming {
    float setup_ms = 0.0f;   // RNG state initialization kernel(s)
    float kernel_ms = 0.0f;  // pricing + reduction kernel(s), summed over batches
};

// What the kernels actually need, precomputed on the host in double and then
// narrowed to float. FP32 because consumer GPUs run FP64 at a small fraction of
// FP32 rate (1/32 on Turing GeForce); precomputing drift/vol means the kernel
// does one exp per path and never touches r, sigma or T.
struct KernelParams {
    float S0;
    float K;
    float drift;  // (r - 0.5*sigma^2) * T
    float vol;    // sigma * sqrt(T)
    int is_call;  // int, not bool: kernel-argument layout identical on host/device
};

inline KernelParams make_kernel_params(const OptionParams& p) {
    return {static_cast<float>(p.S0), static_cast<float>(p.K),
            static_cast<float>((p.r - 0.5 * p.sigma * p.sigma) * p.T),
            static_cast<float>(p.sigma * std::sqrt(p.T)), p.is_call ? 1 : 0};
}

// Both return {NAN, NAN} until their kernels are implemented.
// `timing` may be null.
PriceResult price_gpu_naive(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed,
                            GpuTiming* timing = nullptr);
PriceResult price_gpu_opt(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed,
                          GpuTiming* timing = nullptr);
