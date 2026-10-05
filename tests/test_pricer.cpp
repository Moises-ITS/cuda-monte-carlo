// Minimal test runner. Not assert(): Release builds define NDEBUG, which would
// silently compile every check away.

#include <cmath>
#include <cstdio>
#include <initializer_list>

#include <cuda_runtime.h>

#include "black_scholes.hpp"
#include "cpu_pricer.hpp"
#include "gpu_pricer.cuh"
#include "option.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// 3 SE is a ~99.7% interval: a correct pricer fails this ~0.3% of the time for
// a random seed. Seeds here are fixed, so a pass is reproducible, but if you
// change a seed and it fails *once*, try a few seeds before hunting a bug.
void check_mc(const char* name, PriceResult r, double bs) {
    const double z = (r.price - bs) / r.std_error;
    std::printf("       %-12s price=%.6f se=%.2e bs=%.6f err/SE=%+.2f\n", name, r.price,
                r.std_error, bs, z);
    char what[128];
    std::snprintf(what, sizeof(what), "%s within 3 SE of Black-Scholes", name);
    check(std::isfinite(z) && std::fabs(z) < 3.0, what);
}

void test_black_scholes_known_values() {
    const OptionParams c = default_params(true);
    const OptionParams p = default_params(false);
    // Rounded to 4 decimals, so the tolerance is half a unit in the 4th place.
    check(std::fabs(bs_price(c) - 10.4506) < 5e-5, "BS call = 10.4506");
    check(std::fabs(bs_price(p) - 5.5735) < 5e-5, "BS put  = 5.5735");
}

void test_put_call_parity() {
    // C - P = S0 - K e^{-rT}, model-free. Several regimes: ITM, OTM, short and
    // long maturity, low and high vol.
    const double cases[][5] = {{100, 100, 0.05, 0.2, 1.0},
                               {120, 100, 0.01, 0.4, 0.25},
                               {80, 100, 0.03, 0.1, 5.0},
                               {100, 150, 0.00, 0.6, 2.0}};
    bool ok = true;
    for (const auto& k : cases) {
        const double S0 = k[0], K = k[1], r = k[2], sig = k[3], T = k[4];
        const double lhs = bs_call(S0, K, r, sig, T) - bs_put(S0, K, r, sig, T);
        const double rhs = S0 - K * std::exp(-r * T);
        ok = ok && std::fabs(lhs - rhs) < 1e-10;
    }
    check(ok, "put-call parity (4 parameter sets)");
}

void test_cpu_mc() {
    constexpr std::uint64_t kPaths = 1'000'000;
    constexpr std::uint64_t kSeed = 12345;
    for (const bool is_call : {true, false}) {
        const OptionParams p = default_params(is_call);
        const double bs = bs_price(p);
        std::printf("  %s:\n", is_call ? "call" : "put");
        check_mc("cpu_single", price_cpu_single(p, kPaths, kSeed), bs);
        check_mc("cpu_fast", price_cpu_fast(p, kPaths, kSeed), bs);
        check_mc("cpu_omp", price_cpu_omp(p, kPaths, kSeed), bs);
    }
    // Same seed must give the same answer, otherwise results aren't reproducible.
    const OptionParams p = default_params(true);
    check(price_cpu_fast(p, 10'000, 7).price == price_cpu_fast(p, 10'000, 7).price,
          "cpu_fast deterministic for a fixed seed");
}

void test_gpu_mc() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError();
        std::printf("[SKIP] GPU tests: no CUDA device\n");
        return;
    }
    constexpr std::uint64_t kPaths = 1'000'000;
    const OptionParams p = default_params(true);
    const double bs = bs_price(p);
    const struct {
        const char* name;
        PriceResult (*fn)(const OptionParams&, std::uint64_t, std::uint64_t, GpuTiming*);
    } gpu[] = {{"gpu_naive", price_gpu_naive}, {"gpu_opt", price_gpu_opt}};
    for (const auto& g : gpu) {
        const PriceResult r = g.fn(p, kPaths, 12345, nullptr);
        if (std::isnan(r.price)) {
            std::printf("[SKIP] %s: not implemented yet\n", g.name);
            continue;
        }
        check_mc(g.name, r, bs);
    }
}

}  // namespace

int main() {
    test_black_scholes_known_values();
    test_put_call_parity();
    test_cpu_mc();
    test_gpu_mc();
    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
