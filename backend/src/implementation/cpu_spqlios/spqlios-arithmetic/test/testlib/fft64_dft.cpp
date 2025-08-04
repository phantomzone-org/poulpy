#include "fft64_dft.h"

#include <cstring>

#include "../../spqlios/reim/reim_fft.h"
#include "../../spqlios/reim/reim_fft_internal.h"

reim_fft64vec::reim_fft64vec(uint64_t n) : v(n, 0) {}
reim4_elem reim_fft64vec::get_blk(uint64_t blk) const {
  return reim_view(v.size() / 2, (double*)v.data()).get_blk(blk);
}
double* reim_fft64vec::data() { return v.data(); }
const double* reim_fft64vec::data() const { return v.data(); }
uint64_t reim_fft64vec::nn() const { return v.size(); }
reim_fft64vec::reim_fft64vec(uint64_t n, const double* data) : v(data, data + n) {}
void reim_fft64vec::save_as(double* dest) const { memcpy(dest, v.data(), nn() * sizeof(double)); }
reim_fft64vec reim_fft64vec::zero(uint64_t n) { return reim_fft64vec(n); }
void reim_fft64vec::set_blk(uint64_t blk, const reim4_elem& value) {
  reim_view(v.size() / 2, (double*)v.data()).set_blk(blk, value);
}
reim_fft64vec reim_fft64vec::dft_random(uint64_t n, uint64_t log2bound) {
  return simple_fft64(znx_i64::random_log2bound(n, log2bound));
}
reim_fft64vec reim_fft64vec::random(uint64_t n, double log2bound) {
  double bound = pow(2., log2bound);
  reim_fft64vec res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.v[i] = uniform_f64_bounds(-bound, bound);
  }
  return res;
}

reim_fft64vec operator+(const reim_fft64vec& a, const reim_fft64vec& b) {
  uint64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "ring dimension mismatch");
  reim_fft64vec res(nn);
  double* rv = res.data();
  const double* av = a.data();
  const double* bv = b.data();
  for (uint64_t i = 0; i < nn; ++i) {
    rv[i] = av[i] + bv[i];
  }
  return res;
}
reim_fft64vec operator-(const reim_fft64vec& a, const reim_fft64vec& b) {
  uint64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "ring dimension mismatch");
  reim_fft64vec res(nn);
  double* rv = res.data();
  const double* av = a.data();
  const double* bv = b.data();
  for (uint64_t i = 0; i < nn; ++i) {
    rv[i] = av[i] - bv[i];
  }
  return res;
}
reim_fft64vec operator*(const reim_fft64vec& a, const reim_fft64vec& b) {
  uint64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "ring dimension mismatch");
  REQUIRE_DRAMATICALLY(nn >= 2, "test not defined for nn=1");
  uint64_t m = nn / 2;
  reim_fft64vec res(nn);
  double* rv = res.data();
  const double* av = a.data();
  const double* bv = b.data();
  for (uint64_t i = 0; i < m; ++i) {
    rv[i] = av[i] * bv[i] - av[m + i] * bv[m + i];
    rv[m + i] = av[i] * bv[m + i] + av[m + i] * bv[i];
  }
  return res;
}
reim_fft64vec& operator+=(reim_fft64vec& a, const reim_fft64vec& b) {
  uint64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "ring dimension mismatch");
  double* av = a.data();
  const double* bv = b.data();
  for (uint64_t i = 0; i < nn; ++i) {
    av[i] = av[i] + bv[i];
  }
  return a;
}
reim_fft64vec& operator-=(reim_fft64vec& a, const reim_fft64vec& b) {
  uint64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "ring dimension mismatch");
  double* av = a.data();
  const double* bv = b.data();
  for (uint64_t i = 0; i < nn; ++i) {
    av[i] = av[i] - bv[i];
  }
  return a;
}

reim_fft64vec simple_fft64(const znx_i64& polynomial) {
  const uint64_t nn = polynomial.nn();
  const uint64_t m = nn / 2;
  reim_fft64vec res(nn);
  double* dat = res.data();
  for (uint64_t i = 0; i < nn; ++i) dat[i] = polynomial.get_coeff(i);
  reim_fft_simple(m, dat);
  return res;
}

znx_i64 simple_rint_ifft64(const reim_fft64vec& fftvec) {
  const uint64_t nn = fftvec.nn();
  const uint64_t m = nn / 2;
  std::vector<double> vv(fftvec.data(), fftvec.data() + nn);
  double* v = vv.data();
  reim_ifft_simple(m, v);
  znx_i64 res(nn);
  for (uint64_t i = 0; i < nn; ++i) {
    res.set_coeff(i, rint(v[i] / m));
  }
  return res;
}

rnx_f64 naive_ifft64(const reim_fft64vec& fftvec) {
  const uint64_t nn = fftvec.nn();
  const uint64_t m = nn / 2;
  std::vector<double> vv(fftvec.data(), fftvec.data() + nn);
  double* v = vv.data();
  reim_ifft_simple(m, v);
  rnx_f64 res(nn);
  for (uint64_t i = 0; i < nn; ++i) {
    res.set_coeff(i, v[i] / m);
  }
  return res;
}
double infty_dist(const reim_fft64vec& a, const reim_fft64vec& b) {
  const uint64_t n = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == a.nn(), "dimensions mismatch");
  const double* da = a.data();
  const double* db = b.data();
  double d = 0;
  for (uint64_t i = 0; i < n; ++i) {
    double di = abs(da[i] - db[i]);
    if (di > d) d = di;
  }
  return d;
}

reim_fft64vec simple_fft64(const rnx_f64& polynomial) {
  const uint64_t nn = polynomial.nn();
  const uint64_t m = nn / 2;
  reim_fft64vec res(nn);
  double* dat = res.data();
  for (uint64_t i = 0; i < nn; ++i) dat[i] = polynomial.get_coeff(i);
  reim_fft_simple(m, dat);
  return res;
}

reim_fft64vec operator*(double coeff, const reim_fft64vec& v) {
  const uint64_t nn = v.nn();
  reim_fft64vec res(nn);
  double* rr = res.data();
  const double* vv = v.data();
  for (uint64_t i = 0; i < nn; ++i) rr[i] = coeff * vv[i];
  return res;
}

rnx_f64 simple_ifft64(const reim_fft64vec& v) {
  const uint64_t nn = v.nn();
  const uint64_t m = nn / 2;
  rnx_f64 res(nn);
  double* dat = res.data();
  memcpy(dat, v.data(), nn * sizeof(double));
  reim_ifft_simple(m, dat);
  return res;
}
