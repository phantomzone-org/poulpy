#include "coeffs_arithmetic.h"

#include <assert.h>
#include <memory.h>

/** res = a + b */
EXPORT void znx_add_i64_ref(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = a[i] + b[i];
  }
}
/** res = a - b */
EXPORT void znx_sub_i64_ref(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = a[i] - b[i];
  }
}

EXPORT void znx_negate_i64_ref(uint64_t nn, int64_t* res, const int64_t* a) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = -a[i];
  }
}
EXPORT void znx_copy_i64_ref(uint64_t nn, int64_t* res, const int64_t* a) { memcpy(res, a, nn * sizeof(int64_t)); }

EXPORT void znx_zero_i64_ref(uint64_t nn, int64_t* res) { memset(res, 0, nn * sizeof(int64_t)); }

EXPORT void rnx_divide_by_m_ref(uint64_t n, double m, double* res, const double* a) {
  const double invm = 1. / m;
  for (uint64_t i = 0; i < n; ++i) {
    res[i] = a[i] * invm;
  }
}

EXPORT void rnx_rotate_f64(uint64_t nn, int64_t p, double* res, const double* in) {
  uint64_t a = (-p) & (2 * nn - 1);  // a= (-p) (pos)mod (2*nn)

  if (a < nn) {  // rotate to the left
    uint64_t nma = nn - a;
    // rotate first half
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = in[j + a];
    }
    for (uint64_t j = nma; j < nn; j++) {
      res[j] = -in[j - nma];
    }
  } else {
    a -= nn;
    uint64_t nma = nn - a;
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = -in[j + a];
    }
    for (uint64_t j = nma; j < nn; j++) {
      // rotate first half
      res[j] = in[j - nma];
    }
  }
}

EXPORT void znx_rotate_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in) {
  uint64_t a = (-p) & (2 * nn - 1);  // a= (-p) (pos)mod (2*nn)

  if (a < nn) {  // rotate to the left
    uint64_t nma = nn - a;
    // rotate first half
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = in[j + a];
    }
    for (uint64_t j = nma; j < nn; j++) {
      res[j] = -in[j - nma];
    }
  } else {
    a -= nn;
    uint64_t nma = nn - a;
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = -in[j + a];
    }
    for (uint64_t j = nma; j < nn; j++) {
      // rotate first half
      res[j] = in[j - nma];
    }
  }
}

EXPORT void rnx_mul_xp_minus_one_f64(uint64_t nn, int64_t p, double* res, const double* in) {
  uint64_t a = (-p) & (2 * nn - 1);  // a= (-p) (pos)mod (2*nn)
  if (a < nn) {                      // rotate to the left
    uint64_t nma = nn - a;
    // rotate first half
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = in[j + a] - in[j];
    }
    for (uint64_t j = nma; j < nn; j++) {
      res[j] = -in[j - nma] - in[j];
    }
  } else {
    a -= nn;
    uint64_t nma = nn - a;
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = -in[j + a] - in[j];
    }
    for (uint64_t j = nma; j < nn; j++) {
      // rotate first half
      res[j] = in[j - nma] - in[j];
    }
  }
}

EXPORT void znx_mul_xp_minus_one_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in) {
  uint64_t a = (-p) & (2 * nn - 1);  // a= (-p) (pos)mod (2*nn)
  if (a < nn) {                      // rotate to the left
    uint64_t nma = nn - a;
    // rotate first half
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = in[j + a] - in[j];
    }
    for (uint64_t j = nma; j < nn; j++) {
      res[j] = -in[j - nma] - in[j];
    }
  } else {
    a -= nn;
    uint64_t nma = nn - a;
    for (uint64_t j = 0; j < nma; j++) {
      res[j] = -in[j + a] - in[j];
    }
    for (uint64_t j = nma; j < nn; j++) {
      // rotate first half
      res[j] = in[j - nma] - in[j];
    }
  }
}

EXPORT void znx_mul_xp_minus_one_inplace_i64(uint64_t nn, int64_t p, int64_t* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  uint64_t nb_modif = 0;
  uint64_t j_start = 0;
  while (nb_modif < nn) {
    // follow the cycle that start with j_start
    uint64_t j = j_start;
    int64_t tmp1 = res[j];
    do {
      // find where the value should go, and with which sign
      uint64_t new_j = (j + p) & _2mn;  // mod 2n to get the position and sign
      uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
      // exchange this position with tmp1 (and take care of the sign)
      int64_t tmp2 = res[new_j_n];
      res[new_j_n] = ((new_j < nn) ? tmp1 : -tmp1) - res[new_j_n];
      tmp1 = tmp2;
      // move to the new location, and store the number of items modified
      ++nb_modif;
      j = new_j_n;
    } while (j != j_start);
    // move to the start of the next cycle:
    // we need to find an index that has not been touched yet, and pick it as next j_start.
    // in practice, it is enough to do +1, because the group of rotations is cyclic and 1 is a generator.
    ++j_start;
  }
}

// 0 < p < 2nn
EXPORT void rnx_automorphism_f64(uint64_t nn, int64_t p, double* res, const double* in) {
  res[0] = in[0];
  uint64_t a = 0;
  uint64_t _2mn = 2 * nn - 1;
  for (uint64_t i = 1; i < nn; i++) {
    a = (a + p) & _2mn;  // i*p mod 2n
    if (a < nn) {
      res[a] = in[i];  // res[ip mod 2n] = res[i]
    } else {
      res[a - nn] = -in[i];
    }
  }
}

EXPORT void znx_automorphism_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in) {
  res[0] = in[0];
  uint64_t a = 0;
  uint64_t _2mn = 2 * nn - 1;
  for (uint64_t i = 1; i < nn; i++) {
    a = (a + p) & _2mn;
    if (a < nn) {
      res[a] = in[i];  // res[ip mod 2n] = res[i]
    } else {
      res[a - nn] = -in[i];
    }
  }
}

EXPORT void rnx_rotate_inplace_f64(uint64_t nn, int64_t p, double* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  uint64_t nb_modif = 0;
  uint64_t j_start = 0;
  while (nb_modif < nn) {
    // follow the cycle that start with j_start
    uint64_t j = j_start;
    double tmp1 = res[j];
    do {
      // find where the value should go, and with which sign
      uint64_t new_j = (j + p) & _2mn;  // mod 2n to get the position and sign
      uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
      // exchange this position with tmp1 (and take care of the sign)
      double tmp2 = res[new_j_n];
      res[new_j_n] = (new_j < nn) ? tmp1 : -tmp1;
      tmp1 = tmp2;
      // move to the new location, and store the number of items modified
      ++nb_modif;
      j = new_j_n;
    } while (j != j_start);
    // move to the start of the next cycle:
    // we need to find an index that has not been touched yet, and pick it as next j_start.
    // in practice, it is enough to do +1, because the group of rotations is cyclic and 1 is a generator.
    ++j_start;
  }
}

EXPORT void znx_rotate_inplace_i64(uint64_t nn, int64_t p, int64_t* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  uint64_t nb_modif = 0;
  uint64_t j_start = 0;
  while (nb_modif < nn) {
    // follow the cycle that start with j_start
    uint64_t j = j_start;
    int64_t tmp1 = res[j];
    do {
      // find where the value should go, and with which sign
      uint64_t new_j = (j + p) & _2mn;  // mod 2n to get the position and sign
      uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
      // exchange this position with tmp1 (and take care of the sign)
      int64_t tmp2 = res[new_j_n];
      res[new_j_n] = (new_j < nn) ? tmp1 : -tmp1;
      tmp1 = tmp2;
      // move to the new location, and store the number of items modified
      ++nb_modif;
      j = new_j_n;
    } while (j != j_start);
    // move to the start of the next cycle:
    // we need to find an index that has not been touched yet, and pick it as next j_start.
    // in practice, it is enough to do +1, because the group of rotations is cyclic and 1 is a generator.
    ++j_start;
  }
}

EXPORT void rnx_mul_xp_minus_one_inplace_f64(uint64_t nn, int64_t p, double* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  uint64_t nb_modif = 0;
  uint64_t j_start = 0;
  while (nb_modif < nn) {
    // follow the cycle that start with j_start
    uint64_t j = j_start;
    double tmp1 = res[j];
    do {
      // find where the value should go, and with which sign
      uint64_t new_j = (j + p) & _2mn;  // mod 2n to get the position and sign
      uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
      // exchange this position with tmp1 (and take care of the sign)
      double tmp2 = res[new_j_n];
      res[new_j_n] = ((new_j < nn) ? tmp1 : -tmp1) - res[new_j_n];
      tmp1 = tmp2;
      // move to the new location, and store the number of items modified
      ++nb_modif;
      j = new_j_n;
    } while (j != j_start);
    // move to the start of the next cycle:
    // we need to find an index that has not been touched yet, and pick it as next j_start.
    // in practice, it is enough to do +1, because the group of rotations is cyclic and 1 is a generator.
    ++j_start;
  }
}

__always_inline int64_t get_base_k_digit(const int64_t x, const uint64_t base_k) {
  return (x << (64 - base_k)) >> (64 - base_k);
}

__always_inline int64_t get_base_k_carry(const int64_t x, const int64_t digit, const uint64_t base_k) {
  return (x - digit) >> base_k;
}

EXPORT void znx_normalize(uint64_t nn, uint64_t base_k, int64_t* out, int64_t* carry_out, const int64_t* in,
                          const int64_t* carry_in) {
  assert(in);
  if (out != 0) {
    if (carry_in != 0x0 && carry_out != 0x0) {
      // with carry in and carry out is computed
      for (uint64_t i = 0; i < nn; ++i) {
        const int64_t x = in[i];
        const int64_t cin = carry_in[i];

        int64_t digit = get_base_k_digit(x, base_k);
        int64_t carry = get_base_k_carry(x, digit, base_k);
        int64_t digit_plus_cin = digit + cin;
        int64_t y = get_base_k_digit(digit_plus_cin, base_k);
        int64_t cout = carry + get_base_k_carry(digit_plus_cin, y, base_k);

        out[i] = y;
        carry_out[i] = cout;
      }
    } else if (carry_in != 0) {
      // with carry in and carry out is dropped
      for (uint64_t i = 0; i < nn; ++i) {
        const int64_t x = in[i];
        const int64_t cin = carry_in[i];

        int64_t digit = get_base_k_digit(x, base_k);
        int64_t digit_plus_cin = digit + cin;
        int64_t y = get_base_k_digit(digit_plus_cin, base_k);

        out[i] = y;
      }

    } else if (carry_out != 0) {
      // no carry in and carry out is computed
      for (uint64_t i = 0; i < nn; ++i) {
        const int64_t x = in[i];

        int64_t y = get_base_k_digit(x, base_k);
        int64_t cout = get_base_k_carry(x, y, base_k);

        out[i] = y;
        carry_out[i] = cout;
      }

    } else {
      // no carry in and carry out is dropped
      for (uint64_t i = 0; i < nn; ++i) {
        out[i] = get_base_k_digit(in[i], base_k);
      }
    }
  } else {
    assert(carry_out);
    if (carry_in != 0x0) {
      // with carry in and carry out is computed
      for (uint64_t i = 0; i < nn; ++i) {
        const int64_t x = in[i];
        const int64_t cin = carry_in[i];

        int64_t digit = get_base_k_digit(x, base_k);
        int64_t carry = get_base_k_carry(x, digit, base_k);
        int64_t digit_plus_cin = digit + cin;
        int64_t y = get_base_k_digit(digit_plus_cin, base_k);
        int64_t cout = carry + get_base_k_carry(digit_plus_cin, y, base_k);

        carry_out[i] = cout;
      }
    } else {
      // no carry in and carry out is computed
      for (uint64_t i = 0; i < nn; ++i) {
        const int64_t x = in[i];

        int64_t y = get_base_k_digit(x, base_k);
        int64_t cout = get_base_k_carry(x, y, base_k);

        carry_out[i] = cout;
      }
    }
  }
}

void znx_automorphism_inplace_i64(uint64_t nn, int64_t p, int64_t* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  const uint64_t m = nn >> 1;
  // reduce p mod 2n
  p &= _2mn;
  // uint64_t vp = p & _2mn;
  /// uint64_t target_modifs = m >> 1;
  // we proceed by increasing binary valuation
  for (uint64_t binval = 1, vp = p & _2mn, orb_size = m; binval < nn;
       binval <<= 1, vp = (vp << 1) & _2mn, orb_size >>= 1) {
    // In this loop, we are going to treat the orbit of indexes = binval mod 2.binval.
    // At the beginning of this loop we have:
    //   vp = binval * p mod 2n
    //   target_modif = m / binval (i.e. order of the orbit binval % 2.binval)

    // first, handle the orders 1 and 2.
    // if p*binval == binval % 2n: we're done!
    if (vp == binval) return;
    // if p*binval == -binval % 2n: nega-mirror the orbit and all the sub-orbits and exit!
    if (((vp + binval) & _2mn) == 0) {
      for (uint64_t j = binval; j < m; j += binval) {
        int64_t tmp = res[j];
        res[j] = -res[nn - j];
        res[nn - j] = -tmp;
      }
      res[m] = -res[m];
      return;
    }
    // if p*binval == binval + n % 2n: negate the orbit and exit
    if (((vp - binval) & _mn) == 0) {
      for (uint64_t j = binval; j < nn; j += 2 * binval) {
        res[j] = -res[j];
      }
      return;
    }
    // if p*binval == n - binval % 2n: mirror the orbit and continue!
    if (((vp + binval) & _mn) == 0) {
      for (uint64_t j = binval; j < m; j += 2 * binval) {
        int64_t tmp = res[j];
        res[j] = res[nn - j];
        res[nn - j] = tmp;
      }
      continue;
    }
    // otherwise we will follow the orbit cycles,
    // starting from binval and -binval in parallel
    uint64_t j_start = binval;
    uint64_t nb_modif = 0;
    while (nb_modif < orb_size) {
      // follow the cycle that start with j_start
      uint64_t j = j_start;
      int64_t tmp1 = res[j];
      int64_t tmp2 = res[nn - j];
      do {
        // find where the value should go, and with which sign
        uint64_t new_j = (j * p) & _2mn;  // mod 2n to get the position and sign
        uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
        // exchange this position with tmp1 (and take care of the sign)
        int64_t tmp1a = res[new_j_n];
        int64_t tmp2a = res[nn - new_j_n];
        if (new_j < nn) {
          res[new_j_n] = tmp1;
          res[nn - new_j_n] = tmp2;
        } else {
          res[new_j_n] = -tmp1;
          res[nn - new_j_n] = -tmp2;
        }
        tmp1 = tmp1a;
        tmp2 = tmp2a;
        // move to the new location, and store the number of items modified
        nb_modif += 2;
        j = new_j_n;
      } while (j != j_start);
      // move to the start of the next cycle:
      // we need to find an index that has not been touched yet, and pick it as next j_start.
      // in practice, it is enough to do *5, because 5 is a generator.
      j_start = (5 * j_start) & _mn;
    }
  }
}

void rnx_automorphism_inplace_f64(uint64_t nn, int64_t p, double* res) {
  const uint64_t _2mn = 2 * nn - 1;
  const uint64_t _mn = nn - 1;
  const uint64_t m = nn >> 1;
  // reduce p mod 2n
  p &= _2mn;
  // uint64_t vp = p & _2mn;
  /// uint64_t target_modifs = m >> 1;
  // we proceed by increasing binary valuation
  for (uint64_t binval = 1, vp = p & _2mn, orb_size = m; binval < nn;
       binval <<= 1, vp = (vp << 1) & _2mn, orb_size >>= 1) {
    // In this loop, we are going to treat the orbit of indexes = binval mod 2.binval.
    // At the beginning of this loop we have:
    //   vp = binval * p mod 2n
    //   target_modif = m / binval (i.e. order of the orbit binval % 2.binval)

    // first, handle the orders 1 and 2.
    // if p*binval == binval % 2n: we're done!
    if (vp == binval) return;
    // if p*binval == -binval % 2n: nega-mirror the orbit and all the sub-orbits and exit!
    if (((vp + binval) & _2mn) == 0) {
      for (uint64_t j = binval; j < m; j += binval) {
        double tmp = res[j];
        res[j] = -res[nn - j];
        res[nn - j] = -tmp;
      }
      res[m] = -res[m];
      return;
    }
    // if p*binval == binval + n % 2n: negate the orbit and exit
    if (((vp - binval) & _mn) == 0) {
      for (uint64_t j = binval; j < nn; j += 2 * binval) {
        res[j] = -res[j];
      }
      return;
    }
    // if p*binval == n - binval % 2n: mirror the orbit and continue!
    if (((vp + binval) & _mn) == 0) {
      for (uint64_t j = binval; j < m; j += 2 * binval) {
        double tmp = res[j];
        res[j] = res[nn - j];
        res[nn - j] = tmp;
      }
      continue;
    }
    // otherwise we will follow the orbit cycles,
    // starting from binval and -binval in parallel
    uint64_t j_start = binval;
    uint64_t nb_modif = 0;
    while (nb_modif < orb_size) {
      // follow the cycle that start with j_start
      uint64_t j = j_start;
      double tmp1 = res[j];
      double tmp2 = res[nn - j];
      do {
        // find where the value should go, and with which sign
        uint64_t new_j = (j * p) & _2mn;  // mod 2n to get the position and sign
        uint64_t new_j_n = new_j & _mn;   // mod n to get just the position
        // exchange this position with tmp1 (and take care of the sign)
        double tmp1a = res[new_j_n];
        double tmp2a = res[nn - new_j_n];
        if (new_j < nn) {
          res[new_j_n] = tmp1;
          res[nn - new_j_n] = tmp2;
        } else {
          res[new_j_n] = -tmp1;
          res[nn - new_j_n] = -tmp2;
        }
        tmp1 = tmp1a;
        tmp2 = tmp2a;
        // move to the new location, and store the number of items modified
        nb_modif += 2;
        j = new_j_n;
      } while (j != j_start);
      // move to the start of the next cycle:
      // we need to find an index that has not been touched yet, and pick it as next j_start.
      // in practice, it is enough to do *5, because 5 is a generator.
      j_start = (5 * j_start) & _mn;
    }
  }
}
