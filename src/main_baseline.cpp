#include <iostream>
#include <cstdlib>   // malloc, free
#include "linalg.hpp"

using namespace std;

int main() {
    int rows = 3;
    int cols = 4;

    // allocate matrix (row-major) and vectors
    double* A = new double[rows * cols]; // equivalent: double* A = (double*) malloc(rows * cols * sizeof(double));
    double* x = new double[cols];        // equivalent: double* x = (double*) malloc(cols * sizeof(double));
    double* y_row = new double[rows];    
    double* y_col = new double[rows]; 

    // --- basic null checks ---
    if (!A || !x || !y_row || !y_col) {
        cerr << "Allocation failed\n";
        delete[] A;
        delete[] x;
        delete[] y_row;
        delete[] y_col;
        return 1;
    }

    // --- dimension compatibility check example ---
    if (rows <= 0 || cols <= 0) {
        cerr << "Invalid matrix dimensions\n";
        delete[] A;
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

    A[0 * cols + 0] = 1;  A[0 * cols + 1] = 2;
    A[1 * cols + 0] = 3;  A[1 * cols + 1] = 4;
    A[2 * cols + 0] = 5;  A[2 * cols + 1] = 6;

    x[0] = 10;
    x[1] = 20;


    // function calls
    multiply_mv_row_major(A, rows, cols, x, y_row);
    multiply_mv_col_major(A, rows, cols, x, y_col);

    // Expected result
    double expected[3] = {50, 110, 170};
    bool ok_row = true;
    bool ok_col = true;


    for (int i = 0; i < rows; ++i) {
        if (y_row[i] != expected[i]) ok_row = false;
        if (y_col[i] != expected[i]) ok_col = false;
    }

    cout << "Row-major result: ";
    for (int i = 0; i < rows; ++i) cout << y_row[i] << " ";
    cout << (ok_row ? "  [OK]\n" : "  [FAIL]\n");

    cout << "Col-major result: ";
    for (int i = 0; i < rows; ++i) cout << y_col[i] << " ";
    cout << (ok_col ? "  [OK]\n" : "  [FAIL]\n");


    delete[] A;
    delete[] x;
    delete[] y_row;
    delete[] y_col;

    return (ok_row && ok_col) ? 0 : 1;
}