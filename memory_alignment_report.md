# Memory Alignment Analysis Summary

This document examines how aligning matrix and vector memory to CPU cache-line boundaries affects the performance of matrix-vector and matrix-matrix multiplication kernels, and under what conditions that alignment benefit is most pronounced.

## Background

Modern CPUs fetch data in fixed-size blocks called cache lines, typically 64 bytes wide. If an array begins at an address that is not a multiple of 64, an element sitting exactly at a cache-line boundary will span two lines, requiring two fetches to load a single value. Inside a tight inner loop that runs millions of times, this split-line penalty accumulates into a measurable overhead.

Two allocation strategies were compared across four square matrix sizes (128×128, 512×512, 1024×1024, and 2048×2048):

- **Unaligned**: standard `new double[]`, whose starting alignment is implementation-defined and typically 8 or 16 bytes — not guaranteed to fall on a 64-byte boundary.
- **Aligned**: `posix_memalign` with a 64-byte boundary, guaranteeing the first element of every matrix and vector starts exactly at a cache-line boundary.

Both variants were filled with identical random data so that only the memory layout differed between measurements.

## Matrix-Vector Results

The row-major matrix-vector kernel showed the clearest alignment benefit at smaller sizes. In row-major traversal, the inner loop streams consecutively through one full row per outer iteration — a predictable access pattern where a misaligned start address produces an extra cache-line load at the very beginning of each row. With 64-byte alignment that extra load disappears, and every row begins on a clean boundary.

At larger sizes (1024×1024 and above) the alignment gap narrowed. Once the matrix exceeds L2 cache capacity, the dominant cost shifts from boundary-crossing penalties to the raw volume of cache misses, which alignment cannot address. The absolute execution times grow steeply while the alignment delta stays roughly constant, making it a smaller fraction of total runtime.

## Matrix-Matrix Results

The transposed-B matrix-matrix kernel showed a similar but smaller absolute improvement from alignment. Because both A and B_transposed are accessed sequentially in the inner loop, the workload is already cache-friendly; alignment shifts the starting position but cannot change the fact that large matrices will still exceed cache. The benefit was more consistent at 512×512, where a significant portion of the working set fits in L2 and boundary-aligned access allows the CPU's prefetcher to operate on cleaner line boundaries.

## Interaction with Compiler Optimisations

The most significant finding was the amplification of the alignment benefit at `-O3`. Aggressive optimization enables auto-vectorisation — the compiler replaces scalar loops with SIMD instructions (SSE/AVX) that load 16 or 32 bytes at a time. SIMD loads have stricter alignment requirements than scalar loads: an aligned load instruction (`vmovapd`) is both faster and simpler for the hardware than its unaligned counterpart (`vmovupd`). With 64-byte-aligned buffers the compiler can safely emit the aligned form; with unaligned buffers it must fall back to the slower variant, or add runtime alignment checks.

At `-O1`, where vectorization is largely absent, the aligned and unaligned results were close. At `-O3`, the aligned variants consistently outperformed unaligned by a wider margin, confirming that alignment matters when the compiler is doing the most aggressive optimisation.

## Conclusion

Memory alignment is a low-cost, high-leverage optimization. The allocation call itself (`posix_memalign`) is no more expensive than `new[]`, and the alignment guarantee it provides removes a class of hidden penalties that become more significant as compiler optimisations grow more aggressive. For numerical kernels compiled at high optimisation levels and operating on data that fits within the L2 cache, aligning to cache-line boundaries is a reliable and defensible practice.
