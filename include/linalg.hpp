// include/linalg.hpp
#pragma once

void multiply_mv_row_major(const double* A, int rows, int cols,
                           const double* x, double* y);

void multiply_mv_col_major(const double* A, int rows, int cols,
                           const double* x, double* y);

void multiply_mm_naive(const double* A, int rowsA, int colsA,
                       const double* B, int rowsB, int colsB,
                       double* C);

void multiply_mm_transposed_b(const double* A, int rowsA, int colsA,
                              const double* BT, int rowsB, int colsB,
                              double* C);

void multiply_mm_blocked(const double* A, int rowsA, int colsA,
                         const double* B, int rowsB, int colsB,
                         double* C);

                              
// later: optimized variants, e.g.
//void multiply_mm_blocked(...);