// src/main_bench.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include "linalg.hpp"

using namespace std;

struct Stats {
    double mean;
    double stdev;
};

struct Size2D {
    int rows;
    int cols;
};

// Benchmark a matrix–vector multiply function pointer
Stats benchmark_mv(void (*f)(const double*, int, int, const double*, double*),
                   const double* A, int rows, int cols,
                   const double* x, double* y,
                   int runs)
{
    vector<double> times;
    times.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        auto start = chrono::high_resolution_clock::now();
        f(A, rows, cols, x, y);
        auto end = chrono::high_resolution_clock::now();
        double us = chrono::duration_cast<chrono::microseconds>(end - start).count();
        times.push_back(us);
    }

    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        sum += times[i];
    }
    double mean = sum / runs;

    double var = 0.0;
    for (int i = 0; i < runs; ++i) {
        double diff = times[i] - mean;
        var += diff * diff;
    }
    var /= runs;
    double stdev = std::sqrt(var);

    Stats s;
    s.mean = mean;
    s.stdev = stdev;
    return s;
}

// Later add: Stats benchmark_mm(...)


int main() {
    srand(42); // seed for reproducibility

    // --- define all test sizes here ---

    vector<pair<int,int>> sizes;
    sizes.push_back({128, 128});
    sizes.push_back({512, 512});
    sizes.push_back({1024, 1024});
    sizes.push_back({4096, 4096});

    int runs = 5;

    cout << "MV benchmarks (all sizes), row-major vs col-major\n";
    cout << "rows,cols,avg_row(us),stdev_row(us),avg_col(us),stdev_col(us)\n";

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        int rows = sizes[idx].first;
        int cols = sizes[idx].second;

        double* A_row = new double[rows * cols];
        double* x     = new double[cols];
        double* y     = new double[rows];
        double* A_col = new double[rows * cols];

        // Fill A_row and x with simple values
        for (int i = 0; i < rows * cols; ++i) {
            A_row[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0; // random values in [-10, 10]
        }
        for (int j = 0; j < cols; ++j) {
            x[j] = (rand() / (double)RAND_MAX) * 20.0 - 10.0; // random values in [-10, 10]
        }
        // Build column-major version A_col from A_row
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                A_col[j * rows + i] = A_row[i * cols + j];
            }
        }
        
        // 1) benchmark MV row-major vs col-major
        Stats stats_row = benchmark_mv(multiply_mv_row_major,
                                       A_row, rows, cols, x, y, runs);

        Stats stats_col = benchmark_mv(multiply_mv_col_major,
                                       A_col, rows, cols, x, y, runs);

        cout << rows << "," << cols << ","
             << stats_row.mean << "," << stats_row.stdev << ","
             << stats_col.mean << "," << stats_col.stdev << "\n";
        
        delete[] A_row;
        delete[] A_col;
        delete[] x;
        delete[] y;
    }

    // Later: print MM benchmark header and loop sizes again,
    // calling benchmark_mm with your MM functions.


    return 0;
}