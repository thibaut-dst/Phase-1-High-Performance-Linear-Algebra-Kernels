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

    // fill A and x with simple test values (see next section)
    for (int i = 0; i < rows * cols; i++) A[i] = i + 1;
    for (int i = 0; i < cols; i++) x[i] = i + 1;

    // call your functions
    multiply_mv_row_major(A, rows, cols, x, y_row);
    multiply_mv_col_major(A, rows, cols, x, y_col);

    // verify results
    cout << "y_row: ";
    for (int i = 0; i < rows; i++) cout << y_row[i] << " ";
    cout << "\ny_col: ";
    for (int i = 0; i < rows; i++) cout << y_col[i] << " ";
    cout << "\n";

    delete[] A;
    delete[] x;
    delete[] y_row;
    delete[] y_col;

    return 0;
}