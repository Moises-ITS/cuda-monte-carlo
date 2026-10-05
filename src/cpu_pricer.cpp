#include "cpu_pricer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include <omp.h>

namespace {

// Per-path constants hoisted out of the loop: exp/sqrt of option params are
// computed once, leaving one exp() per path.
struct Gbm {
    double S0, K, drift, vol, discount;
    bool is_call;

    explicit Gbm(const OptionParams& p)
        : S0(p.S0),
          K(p.K),
          drift((p.r - 0.5 * p.sigma * p.sigma) * p.T),
          vol(p.sigma * std::sqrt(p.T)),
          discount(std::exp(-p.r * p.T)),
          is_call(p.is_call) {}

    // Undiscounted payoff for one standard-normal draw z.
    double payoff(double z) const {
        const double ST = S0 * std::exp(drift + vol * z);
        return is_call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
    }
};

// xoshiro256** (Blackman & Vigna). Much faster than mt19937_64, 256-bit state,
// and has jump() for carving non-overlapping streams for parallel threads.
class Xoshiro256ss {
public:
    explicit Xoshiro256ss(std::uint64_t seed) {
        // The reference recommends seeding via splitmix64 so that similar seeds
        // (0, 1, 2, ...) still give well-mixed, non-zero states.
        for (auto& w : s_) w = splitmix64(seed);
    }

    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in the open interval (0, 1): top 53 bits plus half an ulp, so it
    // can never be 0 (Box-Muller takes log(u1)).
    double uniform_open() {
        return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53;
    }

    // Advances the state by 2^128 draws. Thread t calls this t times, so each
    // thread gets its own 2^128-long subsequence: no overlap, no correlation.
    void jump() {
        static constexpr std::uint64_t kJump[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                                  0xa9582618e03fc9aa, 0x39abdc4529b1661c};
        std::uint64_t t[4] = {0, 0, 0, 0};
        for (std::uint64_t j : kJump) {
            for (int b = 0; b < 64; ++b) {
                if (j & (std::uint64_t{1} << b)) {
                    for (int i = 0; i < 4; ++i) t[i] ^= s_[i];
                }
                next();
            }
        }
        for (int i = 0; i < 4; ++i) s_[i] = t[i];
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    static std::uint64_t splitmix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    std::uint64_t s_[4];
};

// Box-Muller turns two uniforms into two independent normals. Caching the
// second one halves the log/sqrt/sincos cost per normal.
class BoxMuller {
public:
    double operator()(Xoshiro256ss& rng) {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        constexpr double kTwoPi = 6.283185307179586477;
        const double radius = std::sqrt(-2.0 * std::log(rng.uniform_open()));
        const double theta = kTwoPi * rng.uniform_open();
        spare_ = radius * std::sin(theta);
        has_spare_ = true;
        return radius * std::cos(theta);
    }

private:
    double spare_ = 0.0;
    bool has_spare_ = false;
};

// Sums payoffs for paths [0, n) from one RNG stream. Shared by fast and omp so
// the only difference between them is the parallelism.
void accumulate_fast(const Gbm& g, std::uint64_t n, Xoshiro256ss& rng, double& sum,
                     double& sum_sq) {
    BoxMuller normal;
    double s = 0.0, s2 = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        const double x = g.payoff(normal(rng));
        s += x;
        s2 += x * x;
    }
    sum = s;
    sum_sq = s2;
}

}  // namespace

PriceResult price_cpu_single(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed) {
    const Gbm g(p);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0.0, sum_sq = 0.0;
    for (std::uint64_t i = 0; i < n_paths; ++i) {
        const double x = g.payoff(normal(rng));
        sum += x;
        sum_sq += x * x;
    }
    return make_result(sum, sum_sq, n_paths, g.discount);
}

PriceResult price_cpu_fast(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed) {
    const Gbm g(p);
    Xoshiro256ss rng(seed);
    double sum = 0.0, sum_sq = 0.0;
    accumulate_fast(g, n_paths, rng, sum, sum_sq);
    return make_result(sum, sum_sq, n_paths, g.discount);
}

PriceResult price_cpu_omp(const OptionParams& p, std::uint64_t n_paths, std::uint64_t seed) {
    const Gbm g(p);
    double sum = 0.0, sum_sq = 0.0;

    // No `omp for`: each thread takes one contiguous chunk and its own RNG
    // stream, so the result is deterministic for a given thread count.
    // reduction() gives each thread private copies and adds them once at the
    // end, avoiding false sharing on a shared accumulator.
#pragma omp parallel reduction(+ : sum, sum_sq)
    {
        const auto tid = static_cast<std::uint64_t>(omp_get_thread_num());
        const auto nt = static_cast<std::uint64_t>(omp_get_num_threads());
        const std::uint64_t begin = n_paths * tid / nt;
        const std::uint64_t end = n_paths * (tid + 1) / nt;

        Xoshiro256ss rng(seed);
        for (std::uint64_t k = 0; k < tid; ++k) rng.jump();

        double s = 0.0, s2 = 0.0;
        accumulate_fast(g, end - begin, rng, s, s2);
        sum += s;
        sum_sq += s2;
    }
    return make_result(sum, sum_sq, n_paths, g.discount);
}

int cpu_omp_max_threads() { return omp_get_max_threads(); }
