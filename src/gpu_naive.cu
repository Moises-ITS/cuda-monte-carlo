// Deliberately naive GPU pricer: one thread per path, XORWOW state per thread,
// every payoff written to global memory, then a separate reduction.
// It exists to be measured against gpu_optimized.cu, not to be fast.
//
// WINDOWS TDR WATCHDOG: on a GPU that also drives a display, Windows resets the
// driver if a single kernel runs longer than ~2 s (TdrDelay). You'll see
// "cudaErrorLaunchTimeout" or a black screen flicker. That's why the host code
// below processes paths in batches of kPathsPerBatch; keep each launch well
// under 2 s and shrink the batch if the harness ever reports a timeout.
// (Colab/Linux headless GPUs have no watchdog, but batching costs nothing.)

#include <cmath>
#include <cstdio>

#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "cuda_check.hpp"
#include "gpu_pricer.cuh"
#include "timer.hpp"

namespace {

// TODO(you): flip to true once both kernels below are written.
constexpr bool kImplemented = false;

constexpr int kBlockSize = 256;
// Also bounds memory: curandState (XORWOW) is 48 bytes, so one state per path
// for 1e8 paths would need ~4.8 GB. Batching keeps it to ~200 MB.
constexpr std::uint64_t kPathsPerBatch = std::uint64_t{1} << 22;

struct ToDouble {
    __host__ __device__ double operator()(float x) const { return static_cast<double>(x); }
};
struct SquareAsDouble {
    __host__ __device__ double operator()(float x) const {
        const double d = x;
        return d * d;
    }
};

}  // namespace

// Kernels are at global scope (not in the anonymous namespace) so their names
// are easy to filter on in Nsight Compute: --kernel-name setup_rng_xorwow

__global__ void setup_rng_xorwow(curandState* states, unsigned long long seed,
                                 unsigned long long path_offset, unsigned long long n) {
    // TODO(1): Initialize one XORWOW state per path in this batch.
    //   1. Compute the global thread index i = blockIdx.x * blockDim.x + threadIdx.x.
    //   2. Bounds check: if (i >= n) return. The grid is rounded up to a whole
    //      number of blocks, so the last block has threads with no path.
    //   3. curand_init(seed, path_offset + i, 0, &states[i]).
    //      - Same seed for every thread; the *subsequence* (2nd arg) is what
    //        makes streams independent. Using path_offset + i (the global path
    //        id, not the batch-local i) means batch 2 doesn't replay batch 1.
    //      - Don't "fix" it by using seed + i with subsequence 0: different
    //        seeds give no independence guarantee.
    //   WHY this is slow (and why it's the point of the naive version): for
    //   XORWOW, curand_init with subsequence k does a skip-ahead of k * 2^67
    //   draws. That's a lot of work per thread, which is exactly what the
    //   separately reported setup_ms will show you.
}

__global__ void price_paths_naive(curandState* states, KernelParams p, float* payoffs,
                                  unsigned long long n) {
    // TODO(2): One thread prices one path and stores its payoff.
    //   1. Global index + bounds check, as above.
    //   2. Copy states[i] into a local variable. Working on a register copy
    //      avoids a global-memory round trip on every draw; write it back at
    //      the end if you want the stream to continue (here it isn't reused,
    //      so you can skip the write-back; be ready to explain that choice).
    //   3. float z = curand_normal(&local);   // one N(0,1) draw
    //   4. float ST = p.S0 * expf(p.drift + p.vol * z);
    //      Use expf, not exp: exp(float) may promote to double, and FP64 on
    //      this GPU is slow. (Try __expf later and compare accuracy vs speed.)
    //   5. payoff = p.is_call ? fmaxf(ST - p.K, 0.f) : fmaxf(p.K - ST, 0.f);
    //      Store it *undiscounted*; the host discounts once in make_result().
    //   6. payoffs[i] = payoff;
    //   WHY global memory: this is the "obvious" design (map, then reduce).
    //   It costs n * 4 bytes of DRAM writes plus n * 4 bytes of reads in the
    //   reduction, which the optimized version eliminates entirely.
}

PriceResult price_gpu_naive(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed,
                            GpuTiming* timing) {
    if (!kImplemented) {
        std::fprintf(stderr, "[gpu_naive] not implemented yet (see TODOs in src/gpu_naive.cu)\n");
        return {NAN, NAN};
    }

    const std::uint64_t batch_cap = n_paths < kPathsPerBatch ? n_paths : kPathsPerBatch;
    curandState* d_states = nullptr;
    float* d_payoffs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, batch_cap * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_payoffs, batch_cap * sizeof(float)));

    const KernelParams kp = make_kernel_params(p);
    CudaEventTimer timer;
    float setup_ms = 0.0f, kernel_ms = 0.0f;
    double sum = 0.0, sum_sq = 0.0;

    for (std::uint64_t offset = 0; offset < n_paths; offset += batch_cap) {
        const std::uint64_t n = (n_paths - offset < batch_cap) ? n_paths - offset : batch_cap;
        const unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);

        timer.start();
        setup_rng_xorwow<<<grid, kBlockSize>>>(d_states, seed, offset, n);
        CUDA_CHECK_LAST();
        setup_ms += timer.stop();

        timer.start();
        price_paths_naive<<<grid, kBlockSize>>>(d_states, kp, d_payoffs, n);
        CUDA_CHECK_LAST();
        // Reduction in double: summing millions of floats in float loses digits.
        // Two passes (sum, sum of squares) re-read the array; that's part of
        // what makes this version naive.
        sum += thrust::transform_reduce(thrust::device, d_payoffs, d_payoffs + n, ToDouble{},
                                        0.0, thrust::plus<double>());
        sum_sq += thrust::transform_reduce(thrust::device, d_payoffs, d_payoffs + n,
                                           SquareAsDouble{}, 0.0, thrust::plus<double>());
        kernel_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_payoffs));
    CUDA_CHECK(cudaFree(d_states));

    if (timing) *timing = {setup_ms, kernel_ms};
    return make_result(sum, sum_sq, n_paths, std::exp(-p.r * p.T));
}
