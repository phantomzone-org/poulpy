#include "negacyclic_polynomial_impl.h"

// explicit instantiation
EXPLICIT_INSTANTIATE_POLYNOMIAL(__int128_t);
EXPLICIT_INSTANTIATE_POLYNOMIAL(int64_t);
EXPLICIT_INSTANTIATE_POLYNOMIAL(double);

double infty_dist(const rnx_f64& a, const rnx_f64& b) {
  const uint64_t nn = a.nn();
  const double* aa = a.data();
  const double* bb = b.data();
  double res = 0.;
  for (uint64_t i = 0; i < nn; ++i) {
    double d = fabs(aa[i] - bb[i]);
    if (d > res) res = d;
  }
  return res;
}
