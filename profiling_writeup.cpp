/*
Part 2, Question 5: Profiling

We profiled our benchmarked code using Time Profiler and found that the heaviest stack trace was dominated by multiply_mm_naive, which shows that the main bottleneck in our program was the naive matrix-matrix multiplication kernel itself rather than setup code or printing. Most of the runtime was being spent inside the actual computation, so that is the part of the code that mattered most for optimization.

The naive implementation computes matrix multiplication using:

    C[i,j] = sum from k = 0 to colsA - 1 of A[i,k] * B[k,j]

In our row-major layout, the access to A[i * colsA + k] is contiguous across the row, which is cache-friendly. However, the access to B[k * colsB + j] moves down a column of a row-major matrix, so the inner loop touches memory with a strided pattern. That means worse spatial locality, less effective cache reuse, and more cache misses as the matrix size grows.

The transposed-B version uses the same mathematical operation, but with a precomputed transpose BT:

    C[i,j] = sum from k = 0 to colsA - 1 of A[i,k] * BT[j,k]

In code, this corresponds to accessing BT[j * rowsB + k], which makes the inner loop read data contiguously. That gives better cache locality because both A and BT are accessed row-wise in memory. As a result, the CPU can reuse cache lines more effectively, and the hardware prefetcher can do a better job bringing in useful data.

Our benchmark results matched what we saw in the profiler. For the larger matrix sizes, multiply_mm_transposed_b performed better than multiply_mm_naive. This supports the idea that the main issue in the initial implementation was not the number of arithmetic operations, but the memory access pattern in the innermost loop. Profiling helped us confirm that the naive kernel was the main hotspot, and it guided us toward an optimization that improved cache utilization.

Formula summary:

1. Naive matrix multiplication:

    C[i,j] = sum from k = 0 to colsA - 1 of A[i,k] * B[k,j]

2. Transposed-B matrix multiplication:

    C[i,j] = sum from k = 0 to colsA - 1 of A[i,k] * BT[j,k]


Profiler screenshot reference:
- File included alongside this source file: profiler_screenshot.png
- The screenshot shows Time Profiler identifying multiply_mm_naive as the heaviest stack trace.

"Time Profiler output showing that multiply_mm_naive dominates the heaviest stack trace, confirming that the naive matrix-matrix multiplication kernel is the primary runtime bottleneck."
*/

int main() {
    return 0;
}
