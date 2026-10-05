#include "black_scholes.hpp"

#include <cmath>

double norm_cdf(double x) {
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    constexpr double kInvSqrt2 = 0.70710678118654752440;
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

namespace {

struct D1D2 {
    double d1, d2;
};

D1D2 d1_d2(double S0, double K, double r, double sigma, double T) {
    const double vol_sqrt_t = sigma * std::sqrt(T);
    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    return {d1, d1 - vol_sqrt_t};
}

}  // namespace

double bs_call(double S0, double K, double r, double sigma, double T) {
    const auto [d1, d2] = d1_d2(S0, K, r, sigma, T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double bs_put(double S0, double K, double r, double sigma, double T) {
    // Computed directly rather than via parity so the parity test is a real check.
    const auto [d1, d2] = d1_d2(S0, K, r, sigma, T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1);
}

double bs_price(const OptionParams& p) {
    return p.is_call ? bs_call(p.S0, p.K, p.r, p.sigma, p.T)
                     : bs_put(p.S0, p.K, p.r, p.sigma, p.T);
}
