#include "ntt120_dft.h"

#include "mod_q120.h"

// @brief alternative version of the NTT

/** for all s=k/2^17, root_of_unity(s) = omega_0^k */
static mod_q120 root_of_unity(double s) {
  static mod_q120 omega_2pow17{OMEGA1, OMEGA2, OMEGA3, OMEGA4};
  static double _2pow17 = 1 << 17;
  return pow(omega_2pow17, s * _2pow17);
}
static mod_q120 root_of_unity_inv(double s) {
  static mod_q120 omega_2pow17{OMEGA1, OMEGA2, OMEGA3, OMEGA4};
  static double _2pow17 = 1 << 17;
  return pow(omega_2pow17, -s * _2pow17);
}

/** recursive naive ntt */
static void q120_ntt_naive_rec(uint64_t n, double entry_pwr, mod_q120* data) {
  if (n == 1) return;
  const uint64_t h = n / 2;
  const double s = entry_pwr / 2.;
  mod_q120 om = root_of_unity(s);
  for (uint64_t j = 0; j < h; ++j) {
    mod_q120 om_right = data[h + j] * om;
    data[h + j] = data[j] - om_right;
    data[j] = data[j] + om_right;
  }
  q120_ntt_naive_rec(h, s, data);
  q120_ntt_naive_rec(h, s + 0.5, data + h);
}
static void q120_intt_naive_rec(uint64_t n, double entry_pwr, mod_q120* data) {
  if (n == 1) return;
  const uint64_t h = n / 2;
  const double s = entry_pwr / 2.;
  q120_intt_naive_rec(h, s, data);
  q120_intt_naive_rec(h, s + 0.5, data + h);
  mod_q120 om = root_of_unity_inv(s);
  for (uint64_t j = 0; j < h; ++j) {
    mod_q120 dat_diff = half(data[j] - data[h + j]);
    data[j] = half(data[j] + data[h + j]);
    data[h + j] = dat_diff * om;
  }
}

/** user friendly version */
q120_nttvec simple_ntt120(const znx_i64& polynomial) {
  const uint64_t n = polynomial.nn();
  q120_nttvec res(n);
  for (uint64_t i = 0; i < n; ++i) {
    int64_t xi = polynomial.get_coeff(i);
    res.v[i] = mod_q120(xi, xi, xi, xi);
  }
  q120_ntt_naive_rec(n, 0.5, res.v.data());
  return res;
}

znx_i128 simple_intt120(const q120_nttvec& fftvec) {
  const uint64_t n = fftvec.nn();
  q120_nttvec copy = fftvec;
  znx_i128 res(n);
  q120_intt_naive_rec(n, 0.5, copy.v.data());
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, copy.v[i].to_int128());
  }
  return res;
}
bool operator==(const q120_nttvec& a, const q120_nttvec& b) { return a.v == b.v; }

std::vector<mod_q120> q120_ntt_naive(const std::vector<mod_q120>& x) {
  std::vector<mod_q120> res = x;
  q120_ntt_naive_rec(res.size(), 0.5, res.data());
  return res;
}
q120_nttvec::q120_nttvec(uint64_t n) : v(n) {}
q120_nttvec::q120_nttvec(uint64_t n, const q120b* data) : v(n) {
  int64_t* d = (int64_t*)data;
  for (uint64_t i = 0; i < n; ++i) {
    v[i] = mod_q120::from_q120b(d + 4 * i);
  }
}
q120_nttvec::q120_nttvec(uint64_t n, const q120c* data) : v(n) {
  int64_t* d = (int64_t*)data;
  for (uint64_t i = 0; i < n; ++i) {
    v[i] = mod_q120::from_q120c(d + 4 * i);
  }
}
uint64_t q120_nttvec::nn() const { return v.size(); }
q120_nttvec q120_nttvec::zero(uint64_t n) { return q120_nttvec(n); }
void q120_nttvec::save_as(q120a* dest) const {
  int64_t* const d = (int64_t*)dest;
  const uint64_t n = nn();
  for (uint64_t i = 0; i < n; ++i) {
    v[i].save_as_q120a(d + 4 * i);
  }
}
void q120_nttvec::save_as(q120b* dest) const {
  int64_t* const d = (int64_t*)dest;
  const uint64_t n = nn();
  for (uint64_t i = 0; i < n; ++i) {
    v[i].save_as_q120b(d + 4 * i);
  }
}
void q120_nttvec::save_as(q120c* dest) const {
  int64_t* const d = (int64_t*)dest;
  const uint64_t n = nn();
  for (uint64_t i = 0; i < n; ++i) {
    v[i].save_as_q120c(d + 4 * i);
  }
}
mod_q120 q120_nttvec::get_blk(uint64_t blk) const {
  REQUIRE_DRAMATICALLY(blk < nn(), "blk overflow");
  return v[blk];
}
q120_nttvec q120_nttvec::random(uint64_t n) {
  q120_nttvec res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.v[i] = uniform_q120();
  }
  return res;
}
