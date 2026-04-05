// src/inline_bench.cpp
//
// PURPOSE: Investigates the impact of the `inline` keyword and compiler
// optimization levels on matrix-vector and matrix-matrix multiply performance.
//
// The experiment is run at TWO optimization levels by building twice:
//   g++ -O0 -o inline_bench_O0 src/inline_bench.cpp src/linalg.cpp -I include
//   g++ -O3 -o inline_bench_O3 src/inline_bench.cpp src/linalg.cpp -I include

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include "linalg.hpp"

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTION PAIR
//
// Both functions compute the same thing: acc + a * b
//
// fma_op        — no inline hint.  At -O0 the compiler emits a real CALL
//                 instruction every iteration: push args, jump, execute, return.
//                 For a 2048x2048 MV that inner loop runs ~4 million times,
//                 so 4 million extra calls accumulate.
//
// fma_op_inline — has the `inline` hint.  The compiler pastes the body
//                 directly at each call site instead of jumping to a separate
//                 function.  No CALL overhead, no register saves/restores.
//                 At -O3 both behave identically because the compiler inlines
//                 small functions automatically regardless of the keyword.
// ─────────────────────────────────────────────────────────────────────────────
static double fma_op(double acc, double a, double b) {
    return acc + a * b;
}

static inline double fma_op_inline(double acc, double a, double b) {
    return acc + a * b;
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL VARIANTS — MV (matrix-vector)
//
// Both kernels are identical except for which helper they call.
// `static` keeps them file-private, don't need to appear in linalg.hpp.
// `const double*` on inputs signals to the compiler these won't be written,
// which can unlock additional optimizations.
// ─────────────────────────────────────────────────────────────────────────────

// MV row-major using the non-inline helper
static void mv_row_noinline(const double* A, int rows, int cols,
                             const double* x, double* y)
{
    for (int i = 0; i < rows; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < cols; ++j)
            y[i] = fma_op(y[i], A[i * cols + j], x[j]);
    }
}

// MV row-major using the inline helper
static void mv_row_inlined(const double* A, int rows, int cols,
                            const double* x, double* y)
{
    for (int i = 0; i < rows; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < cols; ++j)
            y[i] = fma_op_inline(y[i], A[i * cols + j], x[j]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL VARIANTS — MM (matrix-matrix, transposed B)
//
// /*rowsB*/ comments out the parameter name while keeping the type in the
// signature.  The value is received but not used (colsA == rowsB for square
// matrices).  Commenting out the name suppresses the "unused parameter" warning
// without changing the function signature that bench_mm requires.
// ─────────────────────────────────────────────────────────────────────────────

// MM transposed-B using the non-inline helper
static void mm_trans_noinline(const double* A, int rowsA, int colsA,
                               const double* BT, int /*rowsB*/, int colsB,
                               double* C)
{
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k)
                C[i * colsB + j] = fma_op(C[i * colsB + j],
                                           A[i * colsA + k],
                                           BT[j * colsA + k]);
        }
}

// MM transposed-B using the inline helper
static void mm_trans_inlined(const double* A, int rowsA, int colsA,
                              const double* BT, int /*rowsB*/, int colsB,
                              double* C)
{
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k)
                C[i * colsB + j] = fma_op_inline(C[i * colsB + j],
                                                   A[i * colsA + k],
                                                   BT[j * colsA + k]);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// STATS STRUCT + compute_stats
// Identical to alignment_bench.cpp and main_bench.cpp
// ─────────────────────────────────────────────────────────────────────────────
struct Stats { double mean, stdev; };

static Stats compute_stats(const vector<double>& t) {
    double sum = 0.0;
    for (double v : t) sum += v;
    double mean = sum / t.size();

    double var = 0.0;
    for (double v : t) { double d = v - mean; var += d * d; }
    var /= t.size();

    Stats s; s.mean = mean; s.stdev = sqrt(var);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// bench_mv / bench_mm — timing wrappers with function pointers
//
// A function pointer lets us pass any matching function as an argument.
// One timing loop serves both noinline and inline variants
// without duplicating the chrono boilerplate.
// ─────────────────────────────────────────────────────────────────────────────
static Stats bench_mv(void (*f)(const double*, int, int, const double*, double*),
                      const double* A, int rows, int cols,
                      const double* x, double* y, int runs)
{
    vector<double> times; times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = chrono::high_resolution_clock::now();
        f(A, rows, cols, x, y);
        auto t1 = chrono::high_resolution_clock::now();
        times.push_back(
            chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    }
    return compute_stats(times);
}

static Stats bench_mm(void (*f)(const double*, int, int, const double*, int, int, double*),
                      const double* A, int rA, int cA,
                      const double* B, int rB, int cB,
                      double* C, int runs)
{
    vector<double> times; times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = chrono::high_resolution_clock::now();
        f(A, rA, cA, B, rB, cB, C);
        auto t1 = chrono::high_resolution_clock::now();
        times.push_back(
            chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    }
    return compute_stats(times);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// Runs two experiments and prints CSV output:
//   1. MV row-major  — noinline vs inline helper
//   2. MM transposed-B — noinline vs inline helper
//
// Comparison is between running inline_bench_O0 and
// inline_bench_O3.  At -O0 the inline version should be noticeably faster.
// At -O3 both should be roughly equal.
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // Fixed seed: ensures identical input data across runs and between the
    // O0 / O3 binaries, so any timing difference is from optimization, not data.
    srand(42);

    const int runs    = 10;
    const int runs_mm = 5;

    vector<pair<int,int>> sizes = {
        {128,  128},
        {512,  512},
        {1024, 1024},
        {2048, 2048}
    };

    // ── EXPERIMENT 1: MV noinline vs inline ──────────────────────────────────
    cout << "MV inline benchmark (row-major), noinline vs inline helper\n";
    cout << "rows,cols,"
         << "avg_noinline(us),stdev_noinline(us),"
         << "avg_inline(us),stdev_inline(us)\n";

    // C++17 structured binding unpacks each pair<int,int> into rows and cols
    for (auto& [rows, cols] : sizes) {
        size_t n = (size_t)rows * cols;  // cast before multiply to avoid overflow

        double* A = new double[n];
        double* x = new double[cols];
        double* y = new double[rows];

        // Random fill in [-10, 10]
        for (size_t i = 0; i < n;   ++i) A[i] = (rand()/(double)RAND_MAX)*20.0-10.0;
        for (int   i = 0; i < cols; ++i) x[i] = (rand()/(double)RAND_MAX)*20.0-10.0;

        Stats s_no = bench_mv(mv_row_noinline, A, rows, cols, x, y, runs);
        Stats s_in = bench_mv(mv_row_inlined,  A, rows, cols, x, y, runs);

        cout << rows << "," << cols << ","
             << s_no.mean << "," << s_no.stdev << ","
             << s_in.mean << "," << s_in.stdev << "\n";

        delete[] A; delete[] x; delete[] y;
    }

    // ── EXPERIMENT 2: MM transposed-B noinline vs inline ─────────────────────
    cout << "\nMM inline benchmark (transposed B), noinline vs inline helper\n";
    cout << "rows,cols,"
         << "avg_noinline(us),stdev_noinline(us),"
         << "avg_inline(us),stdev_inline(us)\n";

    for (auto& [rowsA, colsA] : sizes) {
        int rowsB = colsA, colsB = colsA;  // square case: all dims equal
        size_t nA = (size_t)rowsA*colsA;
        size_t nB = (size_t)rowsB*colsB;
        size_t nC = (size_t)rowsA*colsB;

        double* A  = new double[nA];
        double* BT = new double[nB];
        double* C  = new double[nC];

        for (size_t i = 0; i < nA; ++i) A[i] = (rand()/(double)RAND_MAX)*20.0-10.0;

        // Build BT = transpose(B).
        // Generate a plain B_tmp first, then rearrange elements:
        double* B_tmp = new double[nB];
        for (size_t i = 0; i < nB; ++i) B_tmp[i] = (rand()/(double)RAND_MAX)*20.0-10.0;
        for (int i = 0; i < rowsB; ++i)
            for (int j = 0; j < colsB; ++j)
                BT[j * rowsB + i] = B_tmp[i * colsB + j];
        delete[] B_tmp;  

        Stats s_no = bench_mm(mm_trans_noinline, A, rowsA, colsA, BT, rowsB, colsB, C, runs_mm);
        Stats s_in = bench_mm(mm_trans_inlined,  A, rowsA, colsA, BT, rowsB, colsB, C, runs_mm);

        cout << rowsA << "," << colsB << ","
             << s_no.mean << "," << s_no.stdev << ","
             << s_in.mean << "," << s_in.stdev << "\n";

        delete[] A; delete[] BT; delete[] C;
    }

    return 0;
}
