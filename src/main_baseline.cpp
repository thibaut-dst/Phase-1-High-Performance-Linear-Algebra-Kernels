#include <iostream>
#include <cstdlib>   // malloc, free
#include <chrono>
#include "../include/linalg.hpp"

using namespace std;

int main() {
    
    // =========================================================
    // Part 1A: Matrix-Vector baseline test
    // =========================================================

    int rows = 3;
    int cols = 2;


    // allocate matrix (row-major) and vectors
    double* A_row = new double[rows * cols]; // equivalent: double* A = (double*) malloc(rows * cols * sizeof(double));
    double* A_col = new double[rows * cols];
    double* x = new double[cols];        // equivalent: double* x = (double*) malloc(cols * sizeof(double));
    double* y_row = new double[rows];    
    double* y_col = new double[rows]; 

    // --- basic null checks ---
    if (!A_row || !A_col || !x || !y_row || !y_col) {
        cerr << "Allocation failed\n";
        delete[] A_row;
        delete[] A_col;
        delete[] x;
        delete[] y_row;
        delete[] y_col;
        return 1;
    }

    // --- dimension compatibility check example ---
    if (rows <= 0 || cols <= 0) {
        cerr << "Invalid matrix dimensions\n";
        delete[] A_row;
        delete[] A_col;
        delete[] x;
        delete[] y_row;
        delete[] y_col;
        return 1;
    }

    // Fill A and x with a simple, hand-checkable example:
    //
    // A = [ 1  2
    //       3  4
    //       5  6 ]
    //
    // x = [10, 20]
    //
    // Expected y = A * x:
    // row 0: 1*10 + 2*20 = 50
    // row 1: 3*10 + 4*20 = 110
    // row 2: 5*10 + 6*20 = 170

        // Row-major storage
    A_row[0 * cols + 0] = 1;  A_row[0 * cols + 1] = 2;
    A_row[1 * cols + 0] = 3;  A_row[1 * cols + 1] = 4;
    A_row[2 * cols + 0] = 5;  A_row[2 * cols + 1] = 6;

    // Column-major storage of same matrix
    A_col[0 * rows + 0] = 1;
    A_col[0 * rows + 1] = 3;
    A_col[0 * rows + 2] = 5;
    A_col[1 * rows + 0] = 2;
    A_col[1 * rows + 1] = 4;
    A_col[1 * rows + 2] = 6;

    x[0] = 10;
    x[1] = 20;

    multiply_mv_row_major(A_row, rows, cols, x, y_row);
    multiply_mv_col_major(A_col, rows, cols, x, y_col);


    // Expected result
    double expected[3] = {50, 110, 170};
    bool ok_row = true;
    bool ok_col = true;


    for (int i = 0; i < rows; ++i) {
        if (y_row[i] != expected[i]) ok_row = false;
        if (y_col[i] != expected[i]) ok_col = false;
    }

    cout << "Row-major MV result: ";
    for (int i = 0; i < rows; ++i) cout << y_row[i] << " ";
    cout << (ok_row ? "[OK]\n" : "[FAIL]\n");

    cout << "Col-major MV result: ";
    for (int i = 0; i < rows; ++i) cout << y_col[i] << " ";
    cout << (ok_col ? "[OK]\n" : "[FAIL]\n");

    delete[] A_row;
    delete[] A_col;
    delete[] x;
    delete[] y_row;
    delete[] y_col;

    // =========================================================
    // Part 1B: Matrix-Matrix baseline test
    // =========================================================

    int rowsA = 2, colsA = 2;
    int rowsB = 2, colsB = 2;

    double* M1 = new double[rowsA * colsA];
    double* M2 = new double[rowsB * colsB];
    double* M2T = new double[rowsB * colsB];
    double* C_naive = new double[rowsA * colsB];
    double* C_transposed = new double[rowsA * colsB];

    if (!M1 || !M2 || !M2T || !C_naive || !C_transposed) {
        cerr << "MM allocation failed\n";
        delete[] M1;
        delete[] M2;
        delete[] M2T;
        delete[] C_naive;
        delete[] C_transposed;
        return 1;
    }

    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0 || colsA != rowsB) {
        cerr << "Invalid MM dimensions\n";
        delete[] M1;
        delete[] M2;
        delete[] M2T;
        delete[] C_naive;
        delete[] C_transposed;
        return 1;
    }

    // M1 = [1 2
    //       3 4]
    //
    // M2 = [5 6
    //       7 8]
    //
    // Expected C = M1 * M2 = [19 22
    //                         43 50]

    M1[0 * colsA + 0] = 1;  M1[0 * colsA + 1] = 2;
    M1[1 * colsA + 0] = 3;  M1[1 * colsA + 1] = 4;

    M2[0 * colsB + 0] = 5;  M2[0 * colsB + 1] = 6;
    M2[1 * colsB + 0] = 7;  M2[1 * colsB + 1] = 8;

    // Transpose of M2 in row-major storage:
    // M2T = [5 7
    //        6 8]
    M2T[0 * rowsB + 0] = 5;
    M2T[0 * rowsB + 1] = 7;
    M2T[1 * rowsB + 0] = 6;
    M2T[1 * rowsB + 1] = 8;

    multiply_mm_naive(M1, rowsA, colsA, M2, rowsB, colsB, C_naive);
    multiply_mm_transposed_b(M1, rowsA, colsA, M2T, rowsB, colsB, C_transposed);

    double expected_mm[4] = {19, 22, 43, 50};
    bool ok_mm_naive = true;
    bool ok_mm_transposed = true;

    for (int i = 0; i < rowsA * colsB; ++i) {
        if (C_naive[i] != expected_mm[i]) ok_mm_naive = false;
        if (C_transposed[i] != expected_mm[i]) ok_mm_transposed = false;
    }

    cout << "Naive MM result:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << C_naive[i * colsB + j] << " ";
        }
        cout << "\n";
    }
    cout << (ok_mm_naive ? "[OK]\n" : "[FAIL]\n");

    cout << "Transposed-B MM result:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << C_transposed[i * colsB + j] << " ";
        }
        cout << "\n";
    }
    cout << (ok_mm_transposed ? "[OK]\n" : "[FAIL]\n");

    delete[] M1;
    delete[] M2;
    delete[] M2T;
    delete[] C_naive;
    delete[] C_transposed;

    return (ok_row && ok_col && ok_mm_naive && ok_mm_transposed) ? 0 : 1;

}