#pragma once

#include "option.hpp"

// Standard normal CDF via erfc. erfc (not 1 + erf) keeps full relative
// precision in the far left tail, where 1 + erf(x) would cancel to zero.
double norm_cdf(double x);

double bs_call(double S0, double K, double r, double sigma, double T);
double bs_put(double S0, double K, double r, double sigma, double T);

// Dispatches on p.is_call.
double bs_price(const OptionParams& p);
