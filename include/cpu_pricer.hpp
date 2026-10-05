#pragma once

#include <cstdint>

#include "option.hpp"

// All three use the exact one-step GBM terminal price, so there is no
// time-discretization error: any gap vs Black-Scholes is pure MC noise.

// Baseline: std::mt19937_64 + std::normal_distribution, one thread.
// Note: normal_distribution's algorithm is implementation-defined, so the same
// seed gives different (equally valid) prices on MSVC vs libstdc++.
PriceResult price_cpu_single(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed);

// Same algorithm with a cheaper RNG: xoshiro256** + Box-Muller, one thread.
// Isolates how much of the baseline's cost is the RNG.
PriceResult price_cpu_fast(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed);

// price_cpu_fast across all OpenMP threads, one independent xoshiro stream per
// thread (via jump()), double-precision accumulation, reduction at the end.
PriceResult price_cpu_omp(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed);

int cpu_omp_max_threads();
