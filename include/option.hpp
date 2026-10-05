#pragma once

#include <cmath>
#include <cstdint>

struct OptionParams {
    double S0;     // spot
    double K;      // strike
    double r;      // continuously compounded risk-free rate
    double sigma;  // volatility
    double T;      // maturity in years
    bool is_call;
};

// Reference case used by the harness and tests: BS call ~10.4506, put ~5.5735.
inline OptionParams default_params(bool is_call = true) {
    return {100.0, 100.0, 0.05, 0.2, 1.0, is_call};
}

struct PriceResult {
    double price;      // discounted MC estimate
    double std_error;  // standard error of that estimate
};

// Shared by every implementation so CPU and GPU results are finalized the same
// way. Sums are of *undiscounted* payoffs; discounting once at the end saves a
// multiply per path and keeps the kernels free of r.
inline PriceResult make_result(double sum, double sum_sq, std::uint64_t n, double discount) {
    const double nd = static_cast<double>(n);
    const double mean = sum / nd;
    // Unbiased sample variance. sum_sq/n - mean^2 can cancel catastrophically
    // when variance << mean^2; for vanilla payoffs in double it is fine, but
    // it's the first place to look if the std error ever comes out negative/NaN.
    const double var = (sum_sq / nd - mean * mean) * nd / (nd - 1.0);
    return {discount * mean, discount * std::sqrt(var / nd)};
}
