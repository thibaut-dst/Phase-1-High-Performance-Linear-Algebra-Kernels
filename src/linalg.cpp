#include "linalg.hpp"
#include <iostream>
#include <vector>

using namespace std;

void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result)
{
    if (!matrix || !vector || !result) return;
    if (rows <= 0 || cols <= 0) return;

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
    if (!matrix || !vector || !result) return;
    if (rows <= 0 || cols <= 0) return;

    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[j * rows + i] * vector[j];

        };
    };
};

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result)
{
    if (!matrixA || !matrixB || !result) return;
    if (colsA != rowsB || rowsA <= 0 || colsA <= 0 || colsB <= 0) return;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k) {
                result[i * colsB + j] +=
                    matrixA[i * colsA + k] * matrixB[k * colsB + j];
            };
        };
    };
};

void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result)
{
    // Basic safety checks
    if (!matrixA || !matrixB_transposed || !result) return;
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) return;

    // For C = A (rowsA x colsA) * B (rowsB x colsB),
    // we need colsA == rowsB.
    if (colsA != rowsB) return;

    // Result C is rowsA x colsB, row-major.
    // matrixA is rowsA x colsA, row-major.
    // matrixB_transposed is the transpose of B:
    //   original B is rowsB x colsB, row-major
    //   BT(j, i) = B(i, j) stored as BT[j * rowsB + i]

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                // A(i,k): row-major
                double a_ik = matrixA[i * colsA + k];
                // BT(j,k): row-major, BT[j * rowsB + k]
                double bt_jk = matrixB_transposed[j * rowsB + k];
                sum += a_ik * bt_jk;
            }
            result[i * colsB + j] = sum;
        }
    }
}