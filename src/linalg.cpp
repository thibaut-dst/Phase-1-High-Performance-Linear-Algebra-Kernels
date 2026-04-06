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