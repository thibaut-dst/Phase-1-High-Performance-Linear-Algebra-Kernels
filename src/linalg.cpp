#include <iostream>
#include <vector>

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
            result[i] += matrix[i + j * rows] * vector[j];

        };
    };
};

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result);


void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result);
