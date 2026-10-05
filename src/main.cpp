// Benchmark harness: runs each implementation on the same option, times it,
// and checks the answer against Black-Scholes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "black_scholes.hpp"
#include "cpu_pricer.hpp"
#include "cuda_check.hpp"
#include "gpu_pricer.cuh"
#include "option.hpp"
#include "timer.hpp"

namespace {

// Uniform signature so the harness can loop over implementations. CPU pricers
// ignore the timing out-param; their time is measured with a host clock.
using PriceFn = PriceResult (*)(const OptionParams&, std::uint64_t, std::uint64_t, GpuTiming*);

struct Impl {
    const char* name;
    bool gpu;
    PriceFn fn;
};

const Impl kImpls[] = {
    {"cpu_single", false,
     [](const OptionParams& p, std::uint64_t n, std::uint64_t s, GpuTiming*) {
         return price_cpu_single(p, n, s);
     }},
    {"cpu_fast", false,
     [](const OptionParams& p, std::uint64_t n, std::uint64_t s, GpuTiming*) {
         return price_cpu_fast(p, n, s);
     }},
    {"cpu_omp", false,
     [](const OptionParams& p, std::uint64_t n, std::uint64_t s, GpuTiming*) {
         return price_cpu_omp(p, n, s);
     }},
    {"gpu_naive", true, price_gpu_naive},
    {"gpu_opt", true, price_gpu_opt},
};

struct Args {
    std::vector<std::uint64_t> paths{1'000'000};
    std::uint64_t seed = 42;
    int runs = 5;
    std::string impl = "all";
    std::string csv;  // empty = don't write CSV
    bool put = false;
};

[[noreturn]] void usage(const char* prog, int code) {
    std::printf(
        "Usage: %s [options]\n"
        "  --paths N[,N...]  path counts, scientific notation ok (default 1e6)\n"
        "                    e.g. --paths 1e5,1e6,1e7,1e8\n"
        "  --seed S          RNG seed (default 42)\n"
        "  --runs R          timed runs after 1 warm-up; median reported (default 5)\n"
        "  --impl NAME[,..]  cpu_single|cpu_fast|cpu_omp|gpu_naive|gpu_opt|all (default all)\n"
        "  --csv FILE        append result rows to FILE (e.g. results/out.csv)\n"
        "  --put             price the put instead of the call\n",
        prog);
    std::exit(code);
}

std::vector<std::uint64_t> parse_paths(const std::string& s) {
    std::vector<std::uint64_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // stod so "1e8" works; MC needs n >= 2 for a sample variance.
        const double v = std::stod(tok);
        if (!(v >= 2.0) || v != std::floor(v)) {
            std::fprintf(stderr, "invalid path count: %s\n", tok.c_str());
            std::exit(EXIT_FAILURE);
        }
        out.push_back(static_cast<std::uint64_t>(v));
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], EXIT_FAILURE);
            return argv[++i];
        };
        try {
            if (flag == "--paths") a.paths = parse_paths(value());
            else if (flag == "--seed") a.seed = std::stoull(value());
            else if (flag == "--runs") a.runs = std::max(1, std::stoi(value()));
            else if (flag == "--impl") a.impl = value();
            else if (flag == "--csv") a.csv = value();
            else if (flag == "--put") a.put = true;
            else if (flag == "--help" || flag == "-h") usage(argv[0], EXIT_SUCCESS);
            else {
                std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
                usage(argv[0], EXIT_FAILURE);
            }
        } catch (const std::exception&) {  // stoi/stod on garbage
            std::fprintf(stderr, "bad value for %s\n", flag.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    return a;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Returns the device name, or "" if no usable GPU (GPU impls are then skipped
// instead of aborting, so the CPU baselines still run on a GPU-less box).
std::string print_gpu_info() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError();  // clear the error so later calls start clean
        std::printf("GPU: none detected (GPU implementations will be skipped)\n");
        return "";
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // cudaDeviceProp::clockRate is deprecated (removed in CUDA 13), so the clock
    // comes from the attribute API, which works on every toolkit version.
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    std::printf("GPU: %s | sm_%d%d | %d SMs | %.0f MHz | %.1f GiB\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount, clock_khz / 1000.0,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    return prop.name;
}

struct Measurement {
    PriceResult r{NAN, NAN};
    double time_ms = NAN;   // GPU: kernel-only (CUDA events). CPU: wall clock.
    double wall_ms = NAN;   // end-to-end call, incl. GPU alloc/setup/copies
    double setup_ms = NAN;  // GPU RNG setup only
};

Measurement measure(const Impl& impl, const OptionParams& p, std::uint64_t n,
                    std::uint64_t seed, int runs) {
    Measurement m;
    GpuTiming t;
    // Warm-up: first call pays one-off costs (CUDA context creation, module
    // load, page faults, OpenMP thread pool spin-up) that aren't the algorithm.
    m.r = impl.fn(p, n, seed, &t);
    if (!std::isfinite(m.r.price)) return m;  // stub or broken: don't time it

    std::vector<double> times, walls, setups;
    for (int i = 0; i < runs; ++i) {
        CpuTimer wall;
        m.r = impl.fn(p, n, seed, &t);  // same seed every run: identical work
        const double w = wall.elapsed_ms();
        walls.push_back(w);
        times.push_back(impl.gpu ? t.kernel_ms : w);
        setups.push_back(impl.gpu ? t.setup_ms : NAN);
    }
    // Median, not mean: robust to the occasional OS hiccup or clock ramp.
    m.time_ms = median(times);
    m.wall_ms = median(walls);
    m.setup_ms = impl.gpu ? median(setups) : NAN;
    return m;
}

// "-" for values that don't apply (CPU setup time, speedup without cpu_single).
std::string num_or_dash(double v, const char* fmt) {
    if (!std::isfinite(v)) return "-";
    char buf[32];
    std::snprintf(buf, sizeof(buf), fmt, v);
    return buf;
}

void warn(const char* msg) { std::printf("    WARNING: %s\n", msg); }

void sanity_check(const Impl& impl, const Measurement& m, double bs, double paths_per_sec) {
    const double err = m.r.price - bs;
    if (!std::isfinite(m.r.std_error) || m.r.std_error <= 0.0)
        warn("std error is not positive/finite: check sum_sq accumulation");
    else if (std::fabs(err) > 3.0 * m.r.std_error)
        warn("|error| > 3 std errors: likely a bias (RNG streams, float sum, payoff), "
             "not noise. Expected ~0.3% of the time by chance.");
    if (impl.gpu && paths_per_sec > 30e9)
        warn("> 30B paths/s: suspicious. Dead-code elimination or a missing sync?");
    if (impl.gpu && m.time_ms < 0.1)
        warn("kernel time < 0.1 ms: too short to trust; did the kernel do any work? "
             "Increase --paths.");
}

void append_csv(const std::string& path, const std::string& row) {
    namespace fs = std::filesystem;
    const fs::path fp(path);
    if (fp.has_parent_path()) fs::create_directories(fp.parent_path());
    const bool need_header = !fs::exists(fp) || fs::file_size(fp) == 0;
    std::ofstream out(fp, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "cannot open %s for writing\n", path.c_str());
        return;
    }
    if (need_header)
        out << "impl,device,option,paths,runs,seed,median_ms,wall_ms,setup_ms,paths_per_sec,"
               "price,std_error,bs_price,abs_error,rel_error,error_in_se\n";
    out << row << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const OptionParams p = default_params(!args.put);
    const double bs = bs_price(p);

    std::vector<const Impl*> selected;
    std::stringstream names(args.impl);
    for (std::string name; std::getline(names, name, ',');) {
        const size_t before = selected.size();
        for (const Impl& impl : kImpls)
            if (name == "all" || name == impl.name) selected.push_back(&impl);
        if (selected.size() == before) {
            std::fprintf(stderr, "unknown --impl %s\n", name.c_str());
            usage(argv[0], EXIT_FAILURE);
        }
    }

    std::printf("Option: %s S0=%.2f K=%.2f r=%.4f sigma=%.4f T=%.2f | Black-Scholes = %.6f\n",
                p.is_call ? "call" : "put", p.S0, p.K, p.r, p.sigma, p.T, bs);
    std::printf("CPU: %d OpenMP threads\n", cpu_omp_max_threads());
    const std::string gpu_name = print_gpu_info();
    std::printf("Runs: 1 warm-up + %d timed (median). GPU time = kernel only; "
                "wall = end-to-end call.\n\n",
                args.runs);

    std::printf("%-10s %12s %11s %11s %10s %11s %11s %10s %10s %8s %8s\n", "impl", "paths",
                "time_ms", "wall_ms", "setup_ms", "paths/s", "price", "std_err", "abs_err",
                "err/SE", "speedup");

    for (const std::uint64_t n : args.paths) {
        double cpu_single_ms = NAN;  // speedup baseline, if cpu_single ran for this n
        for (const Impl* impl : selected) {
            if (impl->gpu && gpu_name.empty()) continue;

            const Measurement m = measure(*impl, p, n, args.seed, args.runs);
            if (!std::isfinite(m.r.price)) {
                std::printf("%-10s %12llu  (skipped: not implemented or returned NaN)\n",
                            impl->name, static_cast<unsigned long long>(n));
                continue;
            }

            const double pps = static_cast<double>(n) / (m.time_ms * 1e-3);
            const double abs_err = std::fabs(m.r.price - bs);
            const double rel_err = abs_err / bs;
            const double err_se = (m.r.price - bs) / m.r.std_error;  // signed
            if (std::string(impl->name) == "cpu_single") cpu_single_ms = m.time_ms;
            const double speedup = cpu_single_ms / m.time_ms;

            std::printf("%-10s %12llu %11.3f %11.3f %10s %11.3e %11.6f %11.3e %10.3e %8.2f %8s\n",
                        impl->name, static_cast<unsigned long long>(n), m.time_ms, m.wall_ms,
                        num_or_dash(m.setup_ms, "%.3f").c_str(), pps, m.r.price, m.r.std_error,
                        abs_err, err_se, num_or_dash(speedup, "%.1f").c_str());
            sanity_check(*impl, m, bs, pps);

            if (!args.csv.empty()) {
                char row[512];
                std::snprintf(row, sizeof(row),
                              "%s,\"%s\",%s,%llu,%d,%llu,%.6f,%.6f,%.6f,%.6e,%.10f,%.6e,%.10f,"
                              "%.6e,%.6e,%.4f",
                              impl->name, impl->gpu ? gpu_name.c_str() : "cpu",
                              p.is_call ? "call" : "put", static_cast<unsigned long long>(n),
                              args.runs, static_cast<unsigned long long>(args.seed), m.time_ms,
                              m.wall_ms, m.setup_ms, pps, m.r.price, m.r.std_error, bs, abs_err,
                              rel_err, err_se);
                append_csv(args.csv, row);
            }
        }
    }
    if (!args.csv.empty()) std::printf("\nAppended results to %s\n", args.csv.c_str());
    return EXIT_SUCCESS;
}
