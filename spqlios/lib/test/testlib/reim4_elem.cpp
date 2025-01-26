#include "reim4_elem.h"

reim4_elem::reim4_elem(const double* re, const double* im) {
  for (uint64_t i = 0; i < 4; ++i) {
    value[i] = re[i];
    value[4 + i] = im[i];
  }
}
reim4_elem::reim4_elem(const double* layout) {
  for (uint64_t i = 0; i < 8; ++i) {
    value[i] = layout[i];
  }
}
reim4_elem::reim4_elem() {
  for (uint64_t i = 0; i < 8; ++i) {
    value[i] = 0.;
  }
}
void reim4_elem::save_re_im(double* re, double* im) const {
  for (uint64_t i = 0; i < 4; ++i) {
    re[i] = value[i];
    im[i] = value[4 + i];
  }
}
void reim4_elem::save_as(double* reim4) const {
  for (uint64_t i = 0; i < 8; ++i) {
    reim4[i] = value[i];
  }
}
reim4_elem reim4_elem::zero() { return reim4_elem(); }

bool operator==(const reim4_elem& x, const reim4_elem& y) {
  for (uint64_t i = 0; i < 8; ++i) {
    if (x.value[i] != y.value[i]) return false;
  }
  return true;
}

reim4_elem gaussian_reim4() {
  test_rng& gen = randgen();
  std::normal_distribution<double> dist(0, 1);
  reim4_elem res;
  for (uint64_t i = 0; i < 8; ++i) {
    res.value[i] = dist(gen);
  }
  return res;
}

reim4_array_view::reim4_array_view(uint64_t size, double* data) : size(size), data(data) {}
reim4_elem reim4_array_view::get(uint64_t i) const {
  REQUIRE_DRAMATICALLY(i < size, "reim4 array overflow");
  return reim4_elem(data + 8 * i);
}
void reim4_array_view::set(uint64_t i, const reim4_elem& value) {
  REQUIRE_DRAMATICALLY(i < size, "reim4 array overflow");
  value.save_as(data + 8 * i);
}

reim_view::reim_view(uint64_t m, double* data) : m(m), data(data) {}
reim4_elem reim_view::get_blk(uint64_t i) {
  REQUIRE_DRAMATICALLY(i < m / 4, "block overflow");
  return reim4_elem(data + 4 * i, data + m + 4 * i);
}
void reim_view::set_blk(uint64_t i, const reim4_elem& value) {
  REQUIRE_DRAMATICALLY(i < m / 4, "block overflow");
  value.save_re_im(data + 4 * i, data + m + 4 * i);
}

reim_vector_view::reim_vector_view(uint64_t m, uint64_t nrows, double* data) : m(m), nrows(nrows), data(data) {}
reim_view reim_vector_view::row(uint64_t row) {
  REQUIRE_DRAMATICALLY(row < nrows, "row overflow");
  return reim_view(m, data + 2 * m * row);
}

/** @brief addition */
reim4_elem operator+(const reim4_elem& x, const reim4_elem& y) {
  reim4_elem reps;
  for (uint64_t i = 0; i < 8; ++i) {
    reps.value[i] = x.value[i] + y.value[i];
  }
  return reps;
}
reim4_elem& operator+=(reim4_elem& x, const reim4_elem& y) {
  for (uint64_t i = 0; i < 8; ++i) {
    x.value[i] += y.value[i];
  }
  return x;
}
/** @brief subtraction */
reim4_elem operator-(const reim4_elem& x, const reim4_elem& y) {
  reim4_elem reps;
  for (uint64_t i = 0; i < 8; ++i) {
    reps.value[i] = x.value[i] + y.value[i];
  }
  return reps;
}
reim4_elem& operator-=(reim4_elem& x, const reim4_elem& y) {
  for (uint64_t i = 0; i < 8; ++i) {
    x.value[i] -= y.value[i];
  }
  return x;
}
/** @brief product */
reim4_elem operator*(const reim4_elem& x, const reim4_elem& y) {
  reim4_elem reps;
  for (uint64_t i = 0; i < 4; ++i) {
    double xre = x.value[i];
    double yre = y.value[i];
    double xim = x.value[i + 4];
    double yim = y.value[i + 4];
    reps.value[i] = xre * yre - xim * yim;
    reps.value[i + 4] = xre * yim + xim * yre;
  }
  return reps;
}
/** @brief distance in infty norm */
double infty_dist(const reim4_elem& x, const reim4_elem& y) {
  double dist = 0;
  for (uint64_t i = 0; i < 8; ++i) {
    double d = fabs(x.value[i] - y.value[i]);
    if (d > dist) dist = d;
  }
  return dist;
}

std::ostream& operator<<(std::ostream& out, const reim4_elem& x) {
  out << "[\n";
  for (uint64_t i = 0; i < 4; ++i) {
    out << "  re=" << x.value[i] << ", im=" << x.value[i + 4] << "\n";
  }
  return out << "]";
}

reim4_matrix_view::reim4_matrix_view(uint64_t nrows, uint64_t ncols, double* data)
    : nrows(nrows), ncols(ncols), data(data) {}
reim4_elem reim4_matrix_view::get(uint64_t row, uint64_t col) const {
  REQUIRE_DRAMATICALLY(row < nrows, "rows out of bounds" << row << " / " << nrows);
  REQUIRE_DRAMATICALLY(col < ncols, "cols out of bounds" << col << " / " << ncols);
  return reim4_elem(data + 8 * (row * ncols + col));
}
void reim4_matrix_view::set(uint64_t row, uint64_t col, const reim4_elem& value) {
  REQUIRE_DRAMATICALLY(row < nrows, "rows out of bounds" << row << " / " << nrows);
  REQUIRE_DRAMATICALLY(col < ncols, "cols out of bounds" << col << " / " << ncols);
  value.save_as(data + 8 * (row * ncols + col));
}
