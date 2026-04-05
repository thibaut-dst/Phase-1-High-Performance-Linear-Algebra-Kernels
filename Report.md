## Cache Locality Analysis

### Matrix–Vector Multiplication: Row-Major vs Column-Major

We implemented two matrix–vector multiplication kernels:

- `multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result)`
- `multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result)`

Both compute \( y = A x \), but they assume different memory layouts for `A`:

- **Row-major layout**: element `(i, j)` is stored at index `i * cols + j`.  
  In our row-major kernel, the inner loop runs over `j` while `i` is fixed, and accesses `matrix[i * cols + j]`. This means it walks linearly through the row in memory. Once a cache line is brought in from RAM, several consecutive elements `matrix[i * cols + j]`, `matrix[i * cols + j+1]`, etc., are likely already sitting in that cache line. This gives strong **spatial locality** and a high cache hit rate.

- **Column-major layout**: element `(i, j)` is stored at index `j * rows + i`.  
  In the column-major kernel we still loop over `j` in the inner loop, but now we access `matrix[j * rows + i]`. From the cache’s perspective, successive accesses in the inner loop are `rows` elements apart in memory. This is a **strided access pattern**: the loop keeps jumping from one row to another instead of walking through contiguous elements. As a result, each cache line we load tends to provide far fewer “useful” elements before the next miss.

In `main_bench.cpp` we construct:

- A row-major matrix `A_row`,
- A column-major copy `A_col` (by transposing the storage layout),
- A compatible vector `x`,

and benchmark both kernels over several sizes (e.g. 128×128, 512×512, 1024×1024, 2048×2048). For each size we run the kernel multiple times and compute the average time and standard deviation using `std::chrono`.

The results show that:

- For small matrices, the difference is modest.
- As matrix size grows, `multiply_mv_row_major` becomes consistently faster than `multiply_mv_col_major`.

This matches the theoretical cache behavior: row-major traversal has better spatial locality and reuses each cache line more effectively than the strided column-major traversal.

---

### Matrix–Matrix Multiplication: Naive vs Transposed B

For matrix–matrix multiplication we benchmark two implementations:

- `multiply_mm_naive(const double* A, int rowsA, int colsA, const double* B, int rowsB, int colsB, double* C)`
- `multiply_mm_transposed_b(const double* A, int rowsA, int colsA, const double* BT, int rowsB, int colsB, double* C)`

Both compute \( C = A \times B \) for compatible sizes (we mostly use square matrices).

In the naive row-major version, the standard triple loop structure is:

\[
C[i,j] = \sum_{k=0}^{colsA-1} A[i,k] \cdot B[k,j].
\]

If `A`, `B`, `C` are stored in row-major order, the access patterns are:

- `A[i,k]` → `A[i * colsA + k]`  
  For fixed `i`, the inner loop over `k` walks across a row of `A`, which is contiguous in memory and cache-friendly.

- `B[k,j]` → `B[k * colsB + j]`  
  For fixed `j`, the inner loop over `k` walks **down a column** of `B`. In row-major storage, the elements of a column are separated by a stride of `colsB`. This is analogous to the column-major MV kernel: we are jumping through memory in large steps, which causes many cache misses and poor spatial locality.

To improve locality for `B`, we precompute its transpose `BT` in row-major layout:

- `BT(j, i) = B(i, j)`, stored as `BT[j * rowsB + i]`.

Inside `multiply_mm_transposed_b`, the innermost loop becomes (conceptually):

\[
C[i,j] = \sum_{k=0}^{colsA-1} A[i,k] \cdot BT[j,k].
\]

Now both accesses in the inner loop are row-wise:

- `A[i,k]` → `A[i * colsA + k]`, row-major contiguous.
- `BT[j,k]` → `BT[j * rowsB + k]`, also row-major contiguous when `k` runs fastest.

This gives good spatial locality for **both** operands. The CPU cache and hardware prefetcher can bring in full cache lines for both `A` and `BT` and reuse them efficiently.

In `main_bench.cpp` we benchmark:

- `multiply_mm_naive` with `A` and `B`, and
- `multiply_mm_transposed_b` with `A` and `BT`,

for a range of square sizes (e.g. 128, 256, 512, 1024). For each size we run multiple iterations and compute average and standard deviation. The timing data shows that the transposed-B version becomes faster than the naive one as matrix size increases, which is consistent with the improved cache utilization for B.

---

### Stride Benchmark: Explicit Cache-Line Demonstration

To make the effect of spatial locality more explicit, we added a separate stride benchmark `stride_bench()`.

- We allocate a large 1D array `arr` of size `N = 8 * 1024 * 1024` doubles (~64 MB).
- For each stride `s` in `{1, 2, 4, 8, 16, 32, 64}`, we repeatedly sum the array elements with:

  ```cpp
  for (int i = 0; i < N; i += s)
      sum += arr[i];
  ```

- We time this loop over several runs and compute the average and standard deviation.
- The program prints CSV-like lines:

  ```text
  Stride benchmarks (sum over big array)
  stride,avg_time(us),stdev_time(us)
  1, ...
  2, ...
  4, ...
  ...
  ```

With stride 1, the loop accesses every element in order, fully exploiting each 64-byte cache line before moving to the next. As the stride increases, we skip more entries in each cache line and jump farther in memory between iterations. The measured times increase with the stride, clearly illustrating how contiguous access (small stride) is much more cache-friendly than strided access.

This is the same phenomenon that explains:

- why row-major MV is faster than column-major MV, and
- why transposing B to improve locality in the inner loop makes our matrix–matrix multiply faster than the naive version.