// Optimized GPU pricer: a fixed grid sized to fill the GPU, Philox RNG, many
// paths per thread, and an in-register -> warp -> block -> global reduction so
// no per-path data ever touches DRAM.
//
// WINDOWS TDR WATCHDOG: Windows resets the driver if one kernel launch on a
// display GPU runs longer than ~2 s. The host loop below splits the work into
// launches of at most kPathsPerLaunch paths. RNG state is saved back to global
// memory at the end of each launch, so batching doesn't change the random
// stream, only how it's chopped up. Measure a single launch and keep it well
// under 2 s; shrink kPathsPerLaunch if the harness reports a timeout.

#include <cmath>
#include <cstdio>

#include <curand_kernel.h>

#include "cuda_check.hpp"
#include "gpu_pricer.cuh"
#include "timer.hpp"

namespace {

// TODO(you): flip to true once both kernels below are written.
constexpr bool kImplemented = false;

// Multiple of 32 (whole warps). 256 is a common starting point; once the
// kernel works, try 128/512 and look at achieved occupancy in Nsight Compute.
constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / 32;
constexpr std::uint64_t kPathsPerLaunch = std::uint64_t{1} << 26;

}  // namespace

__global__ void setup_rng_philox(curandStatePhilox4_32_10_t* states, unsigned long long seed,
                                 int n_threads) {
    // TODO(3): One Philox state per *thread* (not per path).
    //   1. Global index tid; bounds check against n_threads.
    //   2. curand_init(seed, tid, 0, &states[tid]);
    //   WHY Philox: it's counter-based. State is (key, counter), so jumping to
    //   subsequence tid is O(1) arithmetic, unlike XORWOW's skip-ahead. Compare
    //   setup_ms between gpu_naive and gpu_opt and explain the difference.
    //   WHY a separate kernel at all: it lets the harness report RNG setup
    //   separately and lets the state persist across batched launches.
}

__global__ void price_paths_opt(curandStatePhilox4_32_10_t* states, KernelParams p,
                                unsigned long long n_paths, double* d_sums) {
    // d_sums[0] accumulates sum(payoff), d_sums[1] accumulates sum(payoff^2).
    //
    // TODO(4): Per-thread work.
    //   1. tid = blockIdx.x * blockDim.x + threadIdx.x;
    //      stride = gridDim.x * blockDim.x;
    //   2. Load states[tid] into a local variable (registers).
    //   3. Grid-stride loop: for (i = tid; i < n_paths; i += stride)
    //        z = curand_normal(&local); ST = p.S0 * expf(p.drift + p.vol * z);
    //        payoff via fmaxf; acc += payoff; acc_sq += payoff * payoff;
    //      acc/acc_sq are float. WHY a grid-stride loop: the grid is sized to
    //      the hardware (SMs x resident blocks), not to n, so per-thread setup
    //      and the reduction are amortized over many paths.
    //      Later optimization: curand_normal4() returns 4 normals per Philox
    //      call (Philox natively produces 4 x 32 bits); unroll by 4.
    //      Precision: a float sum over thousands of payoffs keeps ~7 digits,
    //      fine here. If your sweep shows drift at huge n, switch acc to double
    //      and measure the cost.
    //   4. Store local back to states[tid] so the next batched launch continues
    //      the stream instead of replaying the same numbers.
    //
    // TODO(5): Warp reduction, no shared memory.
    //   for (int offset = 16; offset > 0; offset >>= 1) {
    //       acc    += __shfl_down_sync(0xffffffff, acc, offset);
    //       acc_sq += __shfl_down_sync(0xffffffff, acc_sq, offset);
    //   }
    //   After this, lane 0 (threadIdx.x % 32 == 0) holds the warp's total.
    //   WHY the full mask is safe: every thread reaches this point (the loop
    //   bound in step 3 doesn't make anyone return early). Don't put a
    //   `return` before the shuffles, or the mask lies and the result is UB.
    //   Decide whether to convert to double before or after this step and be
    //   ready to defend it (double shuffles cost 2x).
    //
    // TODO(6): Block reduction in shared memory.
    //   __shared__ float (or double) warp_sum[kWarpsPerBlock], warp_sq[kWarpsPerBlock];
    //   - lane 0 of each warp writes its total to warp_sum[warp_id].
    //   - __syncthreads();   // all writes visible before anyone reads
    //   - warp 0 loads warp_sum[lane] (0 for lane >= kWarpsPerBlock) and does
    //     the same shuffle reduction again.
    //
    // TODO(7): One global atomic per block.
    //   if (threadIdx.x == 0) {
    //       atomicAdd(&d_sums[0], (double)block_sum);
    //       atomicAdd(&d_sums[1], (double)block_sq);
    //   }
    //   WHY double here: the global total is a sum of billions of payoffs; in
    //   float it would stop changing once it's ~1e7x larger than each addend.
    //   atomicAdd(double*) is native on sm_60+.
    //   Note: atomic order varies run to run, so the last few bits of the price
    //   aren't bit-reproducible. That's expected; know why.
    (void)kWarpsPerBlock;  // silences "unused" until TODO(6) uses it; delete then
}

PriceResult price_gpu_opt(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed,
                          GpuTiming* timing) {
    if (!kImplemented) {
        std::fprintf(stderr, "[gpu_opt] not implemented yet (see TODOs in src/gpu_optimized.cu)\n");
        return {NAN, NAN};
    }

    // Size the grid to exactly what the GPU can keep resident: enough warps to
    // hide latency, no extra blocks queued behind them.
    int device = 0, sm_count = 0, blocks_per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, price_paths_opt,
                                                             kBlockSize, 0));
    const int grid = sm_count * blocks_per_sm;
    const int n_threads = grid * kBlockSize;

    curandStatePhilox4_32_10_t* d_states = nullptr;
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, static_cast<size_t>(n_threads) * sizeof(*d_states)));
    CUDA_CHECK(cudaMalloc(&d_sums, 2 * sizeof(double)));
    // The kernel only ever adds into d_sums, so it must start at zero.
    CUDA_CHECK(cudaMemset(d_sums, 0, 2 * sizeof(double)));

    CudaEventTimer timer;
    timer.start();
    setup_rng_philox<<<grid, kBlockSize>>>(d_states, seed, n_threads);
    CUDA_CHECK_LAST();
    const float setup_ms = timer.stop();

    const KernelParams kp = make_kernel_params(p);
    float kernel_ms = 0.0f;
    for (std::uint64_t offset = 0; offset < n_paths; offset += kPathsPerLaunch) {
        const std::uint64_t n =
            (n_paths - offset < kPathsPerLaunch) ? n_paths - offset : kPathsPerLaunch;
        timer.start();
        price_paths_opt<<<grid, kBlockSize>>>(d_states, kp, n, d_sums);
        CUDA_CHECK_LAST();
        kernel_ms += timer.stop();
    }

    double h_sums[2] = {0.0, 0.0};
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, sizeof(h_sums), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_states));

    if (timing) *timing = {setup_ms, kernel_ms};
    return make_result(h_sums[0], h_sums[1], n_paths, std::exp(-p.r * p.T));
}
