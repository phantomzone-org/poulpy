#ifndef SPQLIOS_NEGACYCLIC_POLYNOMIAL_IMPL_H
#define SPQLIOS_NEGACYCLIC_POLYNOMIAL_IMPL_H

#include "negacyclic_polynomial.h"

template <typename T>
polynomial<T>::polynomial(uint64_t N, const T* c) : coeffs(N) {
  for (uint64_t i = 0; i < N; ++i) coeffs[i] = c[i];
}
/** @brief zero polynomial of dimension N */
template <typename T>
polynomial<T>::polynomial(uint64_t N) : coeffs(N, 0) {}
/** @brief empty polynomial (dim 0) */
template <typename T>
polynomial<T>::polynomial() {}

/** @brief ring dimension */
template <typename T>
uint64_t polynomial<T>::nn() const {
  uint64_t n = coeffs.size();
  REQUIRE_DRAMATICALLY(is_pow2(n), "polynomial dim is not a pow of 2");
  return n;
}

/** @brief special setter (accept any indexes, and does the negacyclic translation) */
template <typename T>
void polynomial<T>::set_coeff(int64_t i, T v) {
  const uint64_t n = nn();
  const uint64_t _2nm = 2 * n - 1;
  uint64_t pos = uint64_t(i) & _2nm;
  if (pos < n) {
    coeffs[pos] = v;
  } else {
    coeffs[pos - n] = -v;
  }
}
/** @brief special getter (accept any indexes, and does the negacyclic translation) */
template <typename T>
T polynomial<T>::get_coeff(int64_t i) const {
  const uint64_t n = nn();
  const uint64_t _2nm = 2 * n - 1;
  uint64_t pos = uint64_t(i) & _2nm;
  if (pos < n) {
    return coeffs[pos];
  } else {
    return -coeffs[pos - n];
  }
}
/** @brief returns the coefficient layout */
template <typename T>
T* polynomial<T>::data() {
  return coeffs.data();
}

template <typename T>
void polynomial<T>::save_as(T* dest) const {
  const uint64_t n = nn();
  for (uint64_t i = 0; i < n; ++i) {
    dest[i] = coeffs[i];
  }
}

/** @brief returns the coefficient layout (const version) */
template <typename T>
const T* polynomial<T>::data() const {
  return coeffs.data();
}

/** @brief returns the coefficient layout (const version) */
template <typename T>
polynomial<T> polynomial<T>::zero(uint64_t n) {
  return polynomial<T>(n);
}

/** @brief equality operator (used during tests) */
template <typename T>
bool operator==(const polynomial<T>& a, const polynomial<T>& b) {
  uint64_t n = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == n, "wrong dimensions");
  for (uint64_t i = 0; i < n; ++i) {
    if (a.get_coeff(i) != b.get_coeff(i)) return false;
  }
  return true;
}

/** @brief addition operator (used during tests) */
template <typename T>
polynomial<T> operator+(const polynomial<T>& a, const polynomial<T>& b) {
  uint64_t n = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == n, "wrong dimensions");
  polynomial<T> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, a.get_coeff(i) + b.get_coeff(i));
  }
  return res;
}

/** @brief subtraction operator (used during tests) */
template <typename T>
polynomial<T> operator-(const polynomial<T>& a, const polynomial<T>& b) {
  uint64_t n = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == n, "wrong dimensions");
  polynomial<T> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, a.get_coeff(i) - b.get_coeff(i));
  }
  return res;
}

/** @brief subtraction operator (used during tests) */
template <typename T>
polynomial<T> operator-(const polynomial<T>& a) {
  uint64_t n = a.nn();
  polynomial<T> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, -a.get_coeff(i));
  }
  return res;
}

/** @brief random polynomial */
template <typename T>
polynomial<T> random_polynomial(uint64_t n);

/** @brief random int64 polynomial */
template <>
polynomial<int64_t> random_polynomial(uint64_t n) {
  polynomial<int64_t> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_i64());
  }
  return res;
}

/** @brief random float64 gaussian polynomial */
template <>
polynomial<double> random_polynomial(uint64_t n) {
  polynomial<double> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, random_f64_gaussian());
  }
  return res;
}

template <typename T>
polynomial<T> random_polynomial_bounds(uint64_t n, const T lb, const T ub);

/** @brief random int64 polynomial */
template <>
polynomial<int64_t> random_polynomial_bounds(uint64_t n, const int64_t lb, const int64_t ub) {
  polynomial<int64_t> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_i64_bounds(lb, ub));
  }
  return res;
}

/** @brief random float64 gaussian polynomial */
template <>
polynomial<double> random_polynomial_bounds(uint64_t n, const double lb, const double ub) {
  polynomial<double> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_f64_bounds(lb, ub));
  }
  return res;
}

/** @brief random int64 polynomial */
template <>
polynomial<__int128_t> random_polynomial_bounds(uint64_t n, const __int128_t lb, const __int128_t ub) {
  polynomial<__int128_t> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_i128_bounds(lb, ub));
  }
  return res;
}

template <typename T>
polynomial<T> random_polynomial_bits(uint64_t n, const uint64_t bits) {
  T b = UINT64_C(1) << bits;
  return random_polynomial_bounds(n, -b, b);
}

template <>
polynomial<int64_t> polynomial<int64_t>::random_log2bound(uint64_t n, uint64_t log2bound) {
  polynomial<int64_t> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_i64_bits(log2bound));
  }
  return res;
}

template <>
polynomial<int64_t> polynomial<int64_t>::random(uint64_t n) {
  polynomial<int64_t> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_u64());
  }
  return res;
}

template <>
polynomial<double> polynomial<double>::random_log2bound(uint64_t n, uint64_t log2bound) {
  polynomial<double> res(n);
  double bound = pow(2., log2bound);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_f64_bounds(-bound, bound));
  }
  return res;
}

template <>
polynomial<double> polynomial<double>::random(uint64_t n) {
  polynomial<double> res(n);
  double bound = 2.;
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, uniform_f64_bounds(-bound, bound));
  }
  return res;
}

template <typename T>
polynomial<T> naive_product(const polynomial<T>& a, const polynomial<T>& b) {
  const int64_t nn = a.nn();
  REQUIRE_DRAMATICALLY(b.nn() == uint64_t(nn), "dimension mismatch!");
  polynomial<T> res(nn);
  for (int64_t i = 0; i < nn; ++i) {
    T ri = 0;
    for (int64_t j = 0; j < nn; ++j) {
      ri += a.get_coeff(j) * b.get_coeff(i - j);
    }
    res.set_coeff(i, ri);
  }
  return res;
}

#define EXPLICIT_INSTANTIATE_POLYNOMIAL(TYPE)                                                    \
  template class polynomial<TYPE>;                                                               \
  template bool operator==(const polynomial<TYPE>& a, const polynomial<TYPE>& b);                \
  template polynomial<TYPE> operator+(const polynomial<TYPE>& a, const polynomial<TYPE>& b);     \
  template polynomial<TYPE> operator-(const polynomial<TYPE>& a, const polynomial<TYPE>& b);     \
  template polynomial<TYPE> operator-(const polynomial<TYPE>& a);                                \
  template polynomial<TYPE> random_polynomial_bits(uint64_t n, const uint64_t bits);             \
  template polynomial<TYPE> naive_product(const polynomial<TYPE>& a, const polynomial<TYPE>& b); \
  // template polynomial<TYPE> random_polynomial(uint64_t n);

#endif  // SPQLIOS_NEGACYCLIC_POLYNOMIAL_IMPL_H
