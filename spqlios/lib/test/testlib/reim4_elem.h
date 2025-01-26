#ifndef SPQLIOS_REIM4_ELEM_H
#define SPQLIOS_REIM4_ELEM_H

#include "test_commons.h"

/** @brief test class representing one single reim4 element */
class reim4_elem {
 public:
  /** @brief 8 components (4 real parts followed by 4 imag parts) */
  double value[8];
  /** @brief constructs from 4 real parts and 4 imaginary parts */
  reim4_elem(const double* re, const double* im);
  /** @brief constructs from 8 components */
  explicit reim4_elem(const double* layout);
  /** @brief zero */
  reim4_elem();
  /** @brief saves the real parts to re and the 4 imag to im */
  void save_re_im(double* re, double* im) const;
  /** @brief saves the 8 components to reim4 */
  void save_as(double* reim4) const;
  static reim4_elem zero();
};

/** @brief checks for equality */
bool operator==(const reim4_elem& x, const reim4_elem& y);
/** @brief random gaussian reim4 of stdev 1 and mean 0 */
reim4_elem gaussian_reim4();
/** @brief addition */
reim4_elem operator+(const reim4_elem& x, const reim4_elem& y);
reim4_elem& operator+=(reim4_elem& x, const reim4_elem& y);
/** @brief subtraction */
reim4_elem operator-(const reim4_elem& x, const reim4_elem& y);
reim4_elem& operator-=(reim4_elem& x, const reim4_elem& y);
/** @brief product */
reim4_elem operator*(const reim4_elem& x, const reim4_elem& y);
std::ostream& operator<<(std::ostream& out, const reim4_elem& x);
/** @brief distance in infty norm */
double infty_dist(const reim4_elem& x, const reim4_elem& y);

/** @brief test class representing the view of one reim of m complexes */
class reim4_array_view {
  uint64_t size;  ///< size of the reim array
  double* data;   ///< pointer to the start of the array
 public:
  /** @brief ininitializes a view at an existing given address */
  reim4_array_view(uint64_t size, double* data);
  ;
  /** @brief gets the i-th element */
  reim4_elem get(uint64_t i) const;
  /** @brief sets the i-th element */
  void set(uint64_t i, const reim4_elem& value);
};

/** @brief test class representing the view of one matrix of nrowsxncols reim4's */
class reim4_matrix_view {
  uint64_t nrows;  ///< number of rows
  uint64_t ncols;  ///< number of columns
  double* data;    ///< pointer to the start of the matrix
 public:
  /** @brief ininitializes a view at an existing given address */
  reim4_matrix_view(uint64_t nrows, uint64_t ncols, double* data);
  /** @brief gets the i-th element */
  reim4_elem get(uint64_t row, uint64_t col) const;
  /** @brief sets the i-th element */
  void set(uint64_t row, uint64_t col, const reim4_elem& value);
};

/** @brief test class representing the view of one reim of m complexes */
class reim_view {
  uint64_t m;    ///< (complex) dimension of the reim polynomial
  double* data;  ///< address of the start of the reim polynomial
 public:
  /** @brief ininitializes a view at an existing given address */
  reim_view(uint64_t m, double* data);
  ;
  /** @brief extracts the i-th reim4 block (i<m/4) */
  reim4_elem get_blk(uint64_t i);
  /** @brief sets the i-th reim4 block (i<m/4) */
  void set_blk(uint64_t i, const reim4_elem& value);
};

/** @brief view of one contiguous reim vector */
class reim_vector_view {
  uint64_t m;      ///< (complex) dimension of the reim polynomial
  uint64_t nrows;  ///< number of reim polynomials
  double* data;    ///< address of the start of the reim polynomial

 public:
  /** @brief ininitializes a view at an existing given address */
  reim_vector_view(uint64_t m, uint64_t nrows, double* data);
  /** @brief view of the given reim */
  reim_view row(uint64_t row);
};

#endif  // SPQLIOS_REIM4_ELEM_H
