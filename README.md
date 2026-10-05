# cuda-mc-pricer

**A CUDA Monte Carlo pricer for European options, benchmarked against CPU baselines and validated against the closed-form Black-Scholes price.**

---

## Why

Monte Carlo pricing is close to the ideal GPU workload. Every simulated path is **independent**: it needs its own random number, a few floating-point operations, and nothing from any other path. The only communication is at the very end, when the payoffs are averaged. That maps directly onto thousands of GPU threads, and the interesting engineering is in what's left over: generating good random numbers in parallel, and reducing billions of payoffs without making memory or atomics the bottleneck.

A fast pricer that returns a wrong number is worse than useless, so this project prices an option that **has a closed-form answer** (Black-Scholes). Every implementation is checked against it, and not just with "close enough". Monte Carlo gives its own error bar (the standard error), so a correct implementation should land within about 3 standard errors of the exact price. Landing outside that range is evidence of a bug (a correlated RNG, a precision loss, a race), not bad luck. The harness reports this for every run.

## Math

Under the risk-neutral measure, the stock follows geometric Brownian motion:

$$dS_t = r S_t\,dt + \sigma S_t\,dW_t$$

This SDE has an exact solution, so for a European option (payoff depends only on $S_T$) we can jump straight to maturity in **one step, with no discretization error**:

$$S_T = S_0 \exp\!\Big(\big(r - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big), \qquad Z \sim \mathcal{N}(0,1)$$

The $-\tfrac{1}{2}\sigma^2$ term is the Itô correction that makes $\mathbb{E}[S_T] = S_0 e^{rT}$.

The option price is the discounted expected payoff:

$$V = e^{-rT}\,\mathbb{E}\big[\,h(S_T)\,\big], \qquad h(S) = \max(S-K,\,0)\ \text{(call)}, \quad \max(K-S,\,0)\ \text{(put)}$$

With $N$ independent draws $Z_i$, the Monte Carlo estimator and its standard error are

$$\hat V_N = e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} h\big(S_T^{(i)}\big), \qquad \mathrm{SE} = \frac{e^{-rT}\,\hat s}{\sqrt{N}}$$

where $\hat s$ is the sample standard deviation of the undiscounted payoffs. By the CLT, $\hat V_N$ is approximately normal around the true price $V$ with standard deviation SE. So $|\hat V_N - V_{BS}| / \mathrm{SE}$ should rarely exceed 3, and the error shrinks only as $1/\sqrt{N}$: **100× more paths for one more correct digit**, which is exactly why throughput matters.

Every implementation accumulates $\sum h$ and $\sum h^2$ and calls the same `make_result()` (in `include/option.hpp`), so they differ only in how they produce those two sums.

Reference case: $S_0=100,\ K=100,\ r=0.05,\ \sigma=0.2,\ T=1$ gives a call of **10.4506** and a put of **5.5735**.

## Implementations

| Implementation | Where | What it demonstrates |
|---|---|---|
| `cpu_single` | `src/cpu_pricer.cpp` | Textbook baseline: `std::mt19937_64` + `std::normal_distribution`, one thread. |
| `cpu_fast` | `src/cpu_pricer.cpp` | Same math, cheaper RNG (xoshiro256\*\* + Box-Muller). Shows how much of the baseline cost is the RNG. |
| `cpu_omp` | `src/cpu_pricer.cpp` | `cpu_fast` on all cores via OpenMP. Independent RNG streams per thread via xoshiro `jump()`, double-precision reduction. The fair CPU comparison point. |
| `gpu_naive` | `src/gpu_naive.cu` | One thread per path, XORWOW `curand_init` per thread, every payoff written to global memory, then a Thrust reduction. Deliberately unoptimized. |
| `gpu_opt` | `src/gpu_optimized.cu` | Philox RNG, occupancy-sized grid with a grid-stride loop, register accumulation, warp-shuffle → shared-memory → one `atomicAdd` per block. No per-path DRAM traffic. |

## Build & run

Requirements: CMake ≥ 3.24, CUDA Toolkit (11.x or 12.x+), a C++17 compiler with OpenMP. Target architecture is `sm_75` (RTX 2060 / T4); override with `-DCMAKE_CUDA_ARCHITECTURES=86` etc. for other GPUs.

### Windows (MSVC + CUDA, native)

Open **"x64 Native Tools Command Prompt for VS 2022"** (it puts `cl.exe` on PATH; a plain terminal won't find it). Ninja ships with Visual Studio.

```bat
cd cuda-mc-pricer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
build\mc_pricer.exe --impl all --paths 1e6
```

Using the Visual Studio generator instead? It's multi-config, so add `--config Release` to the build and run `build\Release\mc_pricer.exe`.

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/mc_pricer --impl all --paths 1e6
```

### Google Colab (T4)

See [`scripts/colab_setup.md`](scripts/colab_setup.md) for the full walkthrough. In short:

```bash
!git clone https://github.com/<you>/cuda-mc-pricer.git
%cd cuda-mc-pricer
!cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
!./build/mc_pricer --impl all --paths 1e6
```

### CLI

```
--paths N[,N...]   path counts; scientific notation ok, e.g. 1e5,1e6,1e7,1e8
--seed S           RNG seed (default 42)
--runs R           timed runs after one warm-up; median reported (default 5)
--impl NAME[,...]  cpu_single | cpu_fast | cpu_omp | gpu_naive | gpu_opt | all
--csv FILE         append one row per (impl, paths) to FILE
--put              price the put instead of the call
```

GPU `time_ms` is **kernel time only** (CUDA events); RNG setup is reported separately as `setup_ms`, and `wall_ms` is the end-to-end call including allocation and copies. The harness prints a warning (without failing) when the error exceeds 3 SE, GPU throughput exceeds 30B paths/s, or kernel time is under 0.1 ms. Those numbers usually mean a bug, not a breakthrough.

A full sweep for the plots:

```bash
./build/mc_pricer --impl all --paths 1e5,1e6,1e7 --csv results/rtx2060.csv
./build/mc_pricer --impl cpu_omp,gpu_naive,gpu_opt --paths 1e8,1e9 --csv results/rtx2060.csv
python scripts/plot_results.py results/rtx2060.csv --out results/rtx2060.png
```

(The second line skips the single-threaded CPU pricers at large sizes, where they take a long time per run.)

## Results

| Implementation | Paths | Time (ms) | Paths/sec | Price | Error vs BS | Speedup |
|---|---|---|---|---|---|---|
| `cpu_single` | TBD | TBD | TBD | TBD | TBD | 1.0× |
| `cpu_fast` | TBD | TBD | TBD | TBD | TBD | TBD |
| `cpu_omp` | TBD | TBD | TBD | TBD | TBD | TBD |
| `gpu_naive` | TBD | TBD | TBD | TBD | TBD | TBD |
| `gpu_opt` | TBD | TBD | TBD | TBD | TBD | TBD |

> Measured on RTX 2060 6GB / Ryzen 5 3600; see `results/` for raw CSVs.
> GPU times are kernel-only; speedup is relative to `cpu_single` at the same path count.

<!-- ![Throughput vs paths](results/rtx2060.png) -->

## Profiling

Nsight Compute (`ncu`) profiles a single kernel launch in detail. Profile one launch of the optimized kernel, skipping the warm-up:

```bash
ncu --kernel-name price_paths_opt --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section LaunchStats \
    -o profiles/gpu_opt ./build/mc_pricer --impl gpu_opt --paths 1e8 --runs 1
```

Open `profiles/gpu_opt.ncu-rep` in the Nsight Compute GUI, or print specific metrics:

```bash
ncu --kernel-name price_paths_opt --launch-skip 1 --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
    ./build/mc_pricer --impl gpu_opt --paths 1e8 --runs 1
```

What to look at:

| Metric | What it tells you |
|---|---|
| **SM throughput** (`sm__throughput`) | How busy the compute units are. This kernel should be compute-bound (RNG + `expf`), so this is the number to push up. |
| **DRAM throughput** (`dram__throughput`) | Should be near zero for `gpu_opt` (no per-path memory traffic) and substantial for `gpu_naive`. That contrast is the whole story of the optimization. |
| **Achieved occupancy** (`sm__warps_active`) | Fraction of warp slots actually in use. Compare to theoretical occupancy (limited by registers per thread; see the Occupancy section). |

Run the same commands with `--kernel-name price_paths_naive` for comparison. The build uses `-lineinfo`, so the Source view maps hot SASS back to `.cu` lines.

On Windows, `ncu` needs GPU performance counters enabled: *NVIDIA Control Panel → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → Allow access to all users*, or run the prompt as Administrator. Otherwise you get `ERR_NVGPUCTRPERM`.

**Findings:** TBD

## Roadmap

- **Antithetic variates**: price with both $Z$ and $-Z$; same RNG cost, lower variance. Report the variance reduction factor.
- **Control variate on $S_T$**: $\mathbb{E}[S_T] = S_0 e^{rT}$ is known exactly; use it to cancel correlated noise.
- **Greeks via pathwise derivatives**: delta and vega from the same paths, validated against closed-form BS Greeks.
- **Arithmetic Asian option (252 steps)**: path-dependent, no closed form; validate against the geometric Asian closed form as a control variate. Makes the kernel genuinely multi-step.
- **FP64 vs FP32**: measure the throughput cost and accuracy gain of double precision on a GeForce part.

## What I learned

TBD
