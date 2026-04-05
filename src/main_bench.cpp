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

Stats benchmark_mm(void (*f)(const double*, int, int, const double*, int, int, double*),
                   const double* A, int rowsA, int colsA,
                   const double* B, int rowsB, int colsB,
                   double* C, int runs)
{
    vector<double> times;
    times.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        auto start = chrono::high_resolution_clock::now();
        f(A, rowsA, colsA, B, rowsB, colsB, C);
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
    double stdev = sqrt(var);

    Stats s;
    s.mean = mean;
    s.stdev = stdev;
    return s;
}

void stride_bench() {
    const int N = 8 * 1024 * 1024; // 8M doubles (~64 MB)
    const int runs = 5;
    double* arr = new double[N];
    for (int i = 0; i < N; ++i) arr[i] = 1.0;

    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    cout << "\nStride benchmarks (sum over big array)\n";
    cout << "stride,avg_time(us),stdev_time(us)\n";

    for (int s = 0; s < num_strides; ++s) {
        int stride = strides[s];
        vector<double> times;
        times.reserve(runs);

        for (int r = 0; r < runs; ++r) {
            auto start = chrono::high_resolution_clock::now();
            volatile double sum = 0.0;  // volatile to avoid being optimized out
            for (int i = 0; i < N; i += stride) {
                sum += arr[i];
            }
            auto end = chrono::high_resolution_clock::now();
            double us = chrono::duration_cast<chrono::microseconds>(end - start).count();
            times.push_back(us);
        }

        double sum_t = 0.0;
        for (int r = 0; r < runs; ++r) sum_t += times[r];
        double mean = sum_t / runs;

        double var = 0.0;
        for (int r = 0; r < runs; ++r) {
            double diff = times[r] - mean;
            var += diff * diff;
        }
        var /= runs;
        double stdev = sqrt(var);

        cout << stride << "," << mean << "," << stdev << "\n";
    }

    delete[] arr;
}

int main() {
    srand(42); // seed for reproducibility

    // --- define all test sizes here ---

    vector<pair<int,int>> sizes;
    sizes.push_back({128, 128});
    sizes.push_back({512, 512});
    sizes.push_back({1024, 1024});
    sizes.push_back({2048, 2048});

    int runs = 5;
    int runs_mm = 3; // MM is much heavier

    // --------------------------------------------------
    // MATRIX-VECTOR BENCHMARKS
    // --------------------------------------------------

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

    // --------------------------------------------------
    // MATRIX-MATRIX BENCHMARKS
    // --------------------------------------------------
    cout << "\nMM benchmarks (all sizes), naive vs transposed B\n";
    cout << "rows,cols,avg_naive(us),stdev_naive(us),avg_transposed(us),stdev_transposed(us)\n";

    // For MM, may want to avoid 4096x4096 if too slow
    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        int rowsA = sizes[idx].first;
        int colsA = sizes[idx].second;

        int rowsB = colsA;          // must be compatible
        int colsB = sizes[idx].second; // square case

        double* A  = new double[rowsA * colsA];
        double* B  = new double[rowsB * colsB];
        double* BT = new double[rowsB * colsB];
        double* C  = new double[rowsA * colsB];

        // Fill A and B with random values
        for (int i = 0; i < rowsA * colsA; ++i) {
            A[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
        }
        for (int i = 0; i < rowsB * colsB; ++i) {
            B[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
        }

        // Build BT = transpose(B), stored in row-major
        // B(i,j)      -> B[i * colsB + j]
        // BT(j,i)     -> BT[j * rowsB + i]
        for (int i = 0; i < rowsB; ++i) {
            for (int j = 0; j < colsB; ++j) {
                BT[j * rowsB + i] = B[i * colsB + j];
            }
        }

        Stats stats_naive = benchmark_mm(multiply_mm_naive,
                                         A, rowsA, colsA,
                                         B, rowsB, colsB,
                                         C, runs_mm);

        Stats stats_transposed = benchmark_mm(multiply_mm_transposed_b,
                                              A, rowsA, colsA,
                                              BT, rowsB, colsB,
                                              C, runs_mm);

        cout << rowsA << "," << colsB << ","
             << stats_naive.mean << "," << stats_naive.stdev << ","
             << stats_transposed.mean << "," << stats_transposed.stdev << "\n";

        delete[] A;
        delete[] B;
        delete[] BT;
        delete[] C;
    }

    stride_bench();  // extra locality experiment

    return 0;
}