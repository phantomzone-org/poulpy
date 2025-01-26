#ifndef SPQLIOS_NEGACYCLIC_POLYNOMIAL_H
#define SPQLIOS_NEGACYCLIC_POLYNOMIAL_H

#include <cstdint>

#include "test_commons.h"

template <typename T>
class polynomial;
typedef polynomial<__int128_t> znx_i128;
typedef polynomial<int64_t> znx_i64;
typedef polynomial<double> rnx_f64;

template <typename T>
class polynomial {
 public:
  std::vector<T> coeffs;
  /** @brief create a polynomial out of existing coeffs */
  polynomial(uint64_t N, const T* c);
  /** @brief zero polynomial of dimension N */
  explicit polynomial(uint64_t N);
  /** @brief empty polynomial (dim 0) */
  polynomial();

  /** @brief ring dimension */
  uint64_t nn() const;
  /** @brief special setter (accept any indexes, and does the negacyclic translation) */
  void set_coeff(int64_t i, T v);
  /** @brief special getter (accept any indexes, and does the negacyclic translation) */
  T get_coeff(int64_t i) const;
  /** @brief returns the coefficient layout */
  T* data();
  /** @brief returns the coefficient layout (const version) */
  const T* data() const;
  /** @brief saves to the layout */
  void save_as(T* dest) const;
  /** @brief zero */
  static polynomial<T> zero(uint64_t n);
  /** @brief random polynomial with coefficients in [-2^log2bounds, 2^log2bounds]*/
  static polynomial<T> random_log2bound(uint64_t n, uint64_t log2bound);
  /** @brief random polynomial with coefficients in [-2^log2bounds, 2^log2bounds]*/
  static polynomial<T> random(uint64_t n);
  /** @brief random polynomial with coefficient in [lb;ub] */
  static polynomial<T> random_bound(uint64_t n, const T lb, const T ub);
};

/** @brief equality operator (used during tests) */
template <typename T>
bool operator==(const polynomial<T>& a, const polynomial<T>& b);

/** @brief addition operator (used during tests) */
template <typename T>
polynomial<T> operator+(const polynomial<T>& a, const polynomial<T>& b);

/** @brief subtraction operator (used during tests) */
template <typename T>
polynomial<T> operator-(const polynomial<T>& a, const polynomial<T>& b);

/** @brief negation operator (used during tests) */
template <typename T>
polynomial<T> operator-(const polynomial<T>& a);

template <typename T>
polynomial<T> naive_product(const polynomial<T>& a, const polynomial<T>& b);

/** @brief distance between two real polynomials (used during tests) */
double infty_dist(const rnx_f64& a, const rnx_f64& b);

#endif  // SPQLIOS_NEGACYCLIC_POLYNOMIAL_H
