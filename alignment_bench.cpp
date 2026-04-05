// src/alignment_bench.cpp
//
// PURPOSE: Measures whether aligning matrix/vector memory to 64-byte boundaries
// (one CPU cache line) produces a measurable speedup over ordinary heap allocation.
//
// HOW TO BUILD:
//   g++ -O1 -o alignment_bench src/alignment_bench.cpp src/linalg.cpp -I include
//   g++ -O3 -o alignment_bench src/alignment_bench.cpp src/linalg.cpp -I include
// Run both and compare

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "linalg.hpp"

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// STATS STRUCT
// Bundles the mean and standard deviation of a set of timing measurements.
// mean  = average run time across all runs (in microseconds)
// stdev = how much runs varied; high stdev = noisy / inconsistent results
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {
    double mean;
    double stdev;
};

// ─────────────────────────────────────────────────────────────────────────────
// compute_stats
// Takes a vector of times and returns mean + stdev.
// ─────────────────────────────────────────────────────────────────────────────
static Stats compute_stats(const vector<double>& times) {
    // Step 1: add all timings, divide by count → mean
    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / times.size();

    // Step 2: variance = average of (each value − mean)²
    double var = 0.0;
    for (double t : times) {
        double d = t - mean;
        var += d * d;
    }
    var /= times.size();

    // Step 3: standard deviation
    Stats s;
    s.mean  = mean;
    s.stdev = sqrt(var);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// alloc_aligned
// Allocates `size` doubles whose start address is a multiple of `align` bytes
// (default 64 = one CPU cache line).
//
// Uses posix_memalign (Linux / macOS) **  MUST be freed with free(), not delete[].
// Mixing allocators is undefined behaviour in C++.
// ─────────────────────────────────────────────────────────────────────────────
static double* alloc_aligned(size_t size, size_t align = 64) {
    void* ptr = nullptr;
    // posix_memalign writes the new address into ptr and returns 0 on success
    if (posix_memalign(&ptr, align, size * sizeof(double)) != 0)
        return nullptr;                    // signal failure to the caller
    return static_cast<double*>(ptr);      // cast generic void* to usable double*
}

// ─────────────────────────────────────────────────────────────────────────────
// alloc_unaligned
// Standard C++ heap allocation via new[].  Alignment is implementation-defined
// Not guaranteed to be 64.  The baseline we compare against alloc_aligned.
// MUST be freed with delete[].
// ─────────────────────────────────────────────────────────────────────────────
static double* alloc_unaligned(size_t size) {
    return new double[size];
}

// ─────────────────────────────────────────────────────────────────────────────
// bench_mv  (matrix-vector timing helper)
// Runs a matrix-vector kernel `runs` times, records each elapsed time in
// microseconds, and returns mean + stdev via compute_stats.
// ─────────────────────────────────────────────────────────────────────────────
static Stats bench_mv(void (*f)(const double*, int, int, const double*, double*),
                      const double* A, int rows, int cols,
                      const double* x, double* y, int runs)
{
    vector<double> times;
    times.reserve(runs);  // pre-allocate so push_back never triggers a resize

    for (int r = 0; r < runs; ++r) {
        auto t0 = chrono::high_resolution_clock::now();  // start timestamp
        f(A, rows, cols, x, y);                          // call whichever kernel was passed
        auto t1 = chrono::high_resolution_clock::now();  // end timestamp

        // duration_cast converts the raw duration object to whole microseconds
        times.push_back(
            chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    }
    return compute_stats(times);
}

// ─────────────────────────────────────────────────────────────────────────────
// bench_mm  (matrix-matrix timing helper)
// Same structure as bench_mv but for matrix-matrix kernels, which take two
// input matrices and their dimensions.  runs_mm is intentionally lower (5)
// because MM is far slower than MV at large sizes.
// ─────────────────────────────────────────────────────────────────────────────
static Stats bench_mm(void (*f)(const double*, int, int, const double*, int, int, double*),
                      const double* A, int rowsA, int colsA,
                      const double* B, int rowsB, int colsB,
                      double* C, int runs)
{
    vector<double> times;
    times.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        auto t0 = chrono::high_resolution_clock::now();
        f(A, rowsA, colsA, B, rowsB, colsB, C);
        auto t1 = chrono::high_resolution_clock::now();
        times.push_back(
            chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    }
    return compute_stats(times);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// Runs two experiments and prints CSV output:
//   1. MV row-major  — aligned vs unaligned across four matrix sizes
//   2. MM transposed-B — aligned vs unaligned across four matrix sizes
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // Fixed seed → rand() produces the same sequence every run.
    // Ensures timing differences come from alignment, not different input data.
    srand(42);

    const int runs    = 10;
    const int runs_mm = 5;

    // Four matrix sizes: small → large.
    vector<pair<int,int>> sizes = {
        {128,  128},
        {512,  512},
        {1024, 1024},
        {2048, 2048}
    };

    // ── EXPERIMENT 1: Matrix-Vector, aligned vs unaligned ────────────────────
    cout << "MV alignment benchmark (row-major), aligned(64B) vs unaligned\n";
    cout << "rows,cols,"
         << "avg_aligned(us),stdev_aligned(us),"
         << "avg_unaligned(us),stdev_unaligned(us)\n";

    // C++17 structured binding: unpacks each pair<int,int> into named variables
    for (auto& [rows, cols] : sizes) {
        // Cast to size_t before multiplying to prevent integer overflow at large sizes
        size_t n = (size_t)rows * cols;

        // Allocate one set with 64-byte alignment, one with plain new[]
        double* A_al = alloc_aligned(n);
        double* x_al = alloc_aligned(cols);
        double* y_al = alloc_aligned(rows);

        double* A_un = alloc_unaligned(n);
        double* x_un = alloc_unaligned(cols);
        double* y_un = alloc_unaligned(rows);

        // Safety: if any alloc returned nullptr skip this size rather than crash
        if (!A_al || !x_al || !y_al || !A_un || !x_un || !y_un) {
            cerr << "Allocation failed for size " << rows << "x" << cols << "\n";
            continue;
        }

        // Fill both sets with the same random values in [-10, 10].
        for (size_t i = 0; i < n;   ++i) A_al[i] = A_un[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
        for (int   i = 0; i < cols; ++i) x_al[i] = x_un[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;

        // Time the same kernel on aligned memory, then on unaligned memory
        Stats s_al = bench_mv(multiply_mv_row_major, A_al, rows, cols, x_al, y_al, runs);
        Stats s_un = bench_mv(multiply_mv_row_major, A_un, rows, cols, x_un, y_un, runs);

        cout << rows << "," << cols << ","
             << s_al.mean << "," << s_al.stdev << ","
             << s_un.mean << "," << s_un.stdev << "\n";

        // Release memory using the matching deallocator for each allocator:
        free(A_al); free(x_al); free(y_al);
        delete[] A_un; delete[] x_un; delete[] y_un;
    }

    // ── EXPERIMENT 2: Matrix-Matrix (transposed B), aligned vs unaligned ─────
    cout << "\nMM alignment benchmark (transposed B), aligned(64B) vs unaligned\n";
    cout << "rows,cols,"
         << "avg_aligned(us),stdev_aligned(us),"
         << "avg_unaligned(us),stdev_unaligned(us)\n";

    for (auto& [rowsA, colsA] : sizes) {
        // Square matrices: rowsB == colsB == colsA
        int rowsB = colsA, colsB = colsA;
        size_t nA = (size_t)rowsA * colsA;  // element count of A
        size_t nB = (size_t)rowsB * colsB;  // element count of B / BT
        size_t nC = (size_t)rowsA * colsB;  // element count of result C

        double* A_al  = alloc_aligned(nA);
        double* BT_al = alloc_aligned(nB);
        double* C_al  = alloc_aligned(nC);

        double* A_un  = alloc_unaligned(nA);
        double* BT_un = alloc_unaligned(nB);
        double* C_un  = alloc_unaligned(nC);

        if (!A_al || !BT_al || !C_al || !A_un || !BT_un || !C_un) {
            cerr << "Allocation failed for size " << rowsA << "x" << colsA << "\n";
            continue;
        }

        // Fill A identically for both variants
        for (size_t i = 0; i < nA; ++i)
            A_al[i] = A_un[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;

        // Build BT (the pre-transposed B that multiply_mm_transposed_b expects).
        // We generate a plain B_tmp first, then swap indices to get the transpose:
        //   B[row i][col j]  lives at  B_tmp[i * colsB + j]
        //   BT[row j][col i] lives at  BT[j * rowsB + i]
        double* B_tmp = alloc_unaligned(nB);
        for (size_t i = 0; i < nB; ++i)
            B_tmp[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;

        for (int i = 0; i < rowsB; ++i)
            for (int j = 0; j < colsB; ++j) {
                double v = B_tmp[i * colsB + j];
                BT_al[j * rowsB + i] = v;  // aligned BT gets the same layout
                BT_un[j * rowsB + i] = v;  // unaligned BT gets the same layout
            }
        delete[] B_tmp;  // temporary buffer no longer needed after transpose

        Stats s_al = bench_mm(multiply_mm_transposed_b,
                              A_al, rowsA, colsA, BT_al, rowsB, colsB, C_al, runs_mm);
        Stats s_un = bench_mm(multiply_mm_transposed_b,
                              A_un, rowsA, colsA, BT_un, rowsB, colsB, C_un, runs_mm);

        cout << rowsA << "," << colsB << ","
             << s_al.mean << "," << s_al.stdev << ","
             << s_un.mean << "," << s_un.stdev << "\n";

        free(A_al); free(BT_al); free(C_al);
        delete[] A_un; delete[] BT_un; delete[] C_un;
    }

    return 0;
}
