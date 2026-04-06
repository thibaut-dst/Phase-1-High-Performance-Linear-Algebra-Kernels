#include "linalg.hpp"
#include <iostream>
#include <vector>

using namespace std;

void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result)
{
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[i * cols + j] * vector[j];

        };
    };
};


void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result)
{
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[j * rows + i] * vector[j];

        };
    };
};

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result)
{
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j) {
            result[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k)
                result[i * colsB + j] += matrixA[i * colsA + k] * matrixB[k * colsB + j];
        }
}

void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                               const double* matrixB_transposed, int rowsB, int colsB,
                               double* result)
{
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j) {
            result[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k)
                result[i * colsB + j] += matrixA[i * colsA + k] * matrixB_transposed[j * colsA + k];
        }
}

void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result)
{
    if (!matrixA || !matrixB || !result) return;
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) return;
    if (colsA != rowsB) return;

    const int block_size = 64;

    for (int i = 0; i < rowsA * colsB; ++i) {
        result[i] = 0.0;
    }

    for (int ii = 0; ii < rowsA; ii += block_size) {
        int i_end = (ii + block_size < rowsA) ? (ii + block_size) : rowsA;

        for (int kk = 0; kk < colsA; kk += block_size) {
            int k_end = (kk + block_size < colsA) ? (kk + block_size) : colsA;

            for (int jj = 0; jj < colsB; jj += block_size) {
                int j_end = (jj + block_size < colsB) ? (jj + block_size) : colsB;

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        double a_ik = matrixA[i * colsA + k];
                        const double* b_row = &matrixB[k * colsB];

                        for (int j = jj; j < j_end; ++j) {
                            result[i * colsB + j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}