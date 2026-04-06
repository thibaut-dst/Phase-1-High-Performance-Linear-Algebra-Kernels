# Phase-1-High-Performance-Linear-Algebra-Kernels

**Objective:** To implement and optimize fundamental linear algebra operations (matrix-vector multiplication and matrix-matrix multiplication) in C++, focusing on performance considerations such as cache locality, memory alignment, and the impact of compiler optimizations like inlining. Teams will analyze the performance of their implementations using benchmarking and profiling tools.


## Team:
- Thibaut Desauty
- Kallubhavi Stephen 
- Abinaya Raut

## Repo structure

```
root/
├─ .gitignore
├─ README.md                # team, build, how to run, answers to discussion Qs
├─ report.pdf               # the formal write‑up
├─ profiling_writeup.cpp
├─ benchmark-results/
│   ├─ benchmarks.csv           # benchmark results (Part 2.1 & 2.2)
│   ├─ alignment_O1.csv         # alignment benchmark results at -O1
│   ├─ alignment_O3.csv         # alignment benchmark results at -O3
│   ├─ inline_O0.csv            # inline benchmark results at -O0
│   └─ inline_O3.csv            # inline benchmark results at -O3
├─ include/
│   └─ linalg.hpp           # function declarations for all kernels
└─ src/
    ├─ linalg.cpp           # implementations of all kernels (Part 1 + optimized)
    ├─ main_baseline.cpp    # simple correctness tests / basic runs
    ├─ main_bench.cpp       # benchmarking for Part 2.1 & 2.2
    ├─ alignment_bench.cpp  # alignment benchmarking (Part 2.3)
    └─ inline_bench.cpp     # inline benchmarking (Part 2.4)
```

## Build instructions

**Part 1 - Correctness tests:**
```bash
g++ -O0 src/main_baseline.cpp src/linalg.cpp -Iinclude -o baseline
./baseline
```

**Part 2.1 & 2.2 - Benchmarking:**
```bash
g++ -O3 src/main_bench.cpp src/linalg.cpp -Iinclude -o bench
./bench > benchmarks.csv
```

**Part 2.3 - Alignment benchmark (unoptimized):**
```bash
g++ -std=c++17 -O1 src/alignment_bench.cpp src/linalg.cpp -I include -o alignment_bench
./alignment_bench > alignment_O1.csv
```

**Part 2.3 - Alignment benchmark (highly optimized, optional):**
```bash
g++ -std=c++17 -O3 src/alignment_bench.cpp src/linalg.cpp -I include -o alignment_bench_O3
./alignment_bench_O3 > alignment_O3.csv
```

**Part 2.4 - Inline benchmark (-O0):**
```bash
g++ -std=c++17 -O0 src/inline_bench.cpp src/linalg.cpp -I include -o inline_bench_O0
./inline_bench_O0 > inline_O0.csv
```

**Part 2.4 - Inline benchmark (-O3):**
```bash
g++ -std=c++17 -O3 src/inline_bench.cpp src/linalg.cpp -I include -o inline_bench_O3
./inline_bench_O3 > inline_O3.csv
```


## Discussion questions

### 1. Pointers vs. References in C++

A reference is an alias to an existing object and cannot be null or reseated. A pointer stores an address, can be null, and can be changed to point somewhere else. In numerical code, I use references for required inputs/outputs and pointers for raw buffers, dynamic memory, and optional values.


### 2. Row-Major vs. Column-Major Storage and Cache Locality

Row-major storage places each row contiguously in memory, so looping across columns gives good spatial locality. Column-major storage places each column contiguously, so accessing a row becomes strided and usually slower in row-major C++ code.

Our benchmarks (`benchmarks.csv`) showed that row-major MV was consistently faster than column-style access, and the gap widened at larger sizes; for 2048×2048, row-major took 2185.6 µs while column-major took 10928.2 µs. Similarly, transposed-B MM was much faster than naive MM for large matrices, showing the benefit of contiguous memory access in the inner loop.


### 3. CPU Caches and Locality

Modern CPUs use a cache hierarchy: L1 is smallest and fastest, then L2, then L3. Spatial locality means nearby data is likely to be reused soon, while temporal locality means recently used data is likely to be used again. Our row-major MV, transposed-B MM, and stride benchmark were designed to exploit or expose these effects.


### 4. Memory Alignment and Performance

Memory alignment means placing data on addresses that match a boundary like 64 bytes. This can reduce cache-line splits and may help SIMD and cache behavior.

The alignment results in `alignment_O1.csv` and `alignment_O3.csv` do not show a consistent win for 64-byte alignment. For MV at -O1, aligned and unaligned times are similar and sometimes the unaligned version is slightly faster (for example, at 1024×1024: 1491.5 µs aligned vs 1178.8 µs unaligned). For MM in the same files, aligned and unaligned versions are also close (for example, at 1024×1024 and -O3: 496371 µs aligned vs 503530 µs unaligned), suggesting that in this code the access pattern and cache locality mattered more than the exact base alignment.


### 5. Compiler Optimizations and Inlining

Inlining removes function-call overhead for small helper functions. At -O0, inline can matter more because the compiler does less automatic optimization. At -O3, the compiler often inlines small functions automatically, so the difference is usually smaller. Aggressive optimization can improve speed, but it can also make debugging harder and sometimes increase code size.

The inline benchmarks in `inline_O0.csv` and `inline_O3.csv` show that the impact of inline is modest and sometimes mixed, and the optimization level matters more than the keyword itself.

**At -O0:** Inlining reduces function-call overhead in some MV cases (e.g., at 1024×1024, MV took 4042.5 µs without inline and 3453.5 µs with inline), but not consistently (at 512×512, inline is actually slightly slower: 1007 µs vs 1114.8 µs). For MM at -O0, the differences are very small compared to the total runtime (e.g., at 512×512: 528080 µs noinline vs 529624 µs inline), because most time is spent doing arithmetic rather than the tiny helper-call overhead.

**At -O3:** The compiler applies aggressive optimizations, so both versions become much faster overall, and the gap between inline and noinline is small and sometimes noisy. For MV at 2048×2048, the inline version is slightly faster (2134.2 µs vs 2442.2 µs), while at 1024×1024 the noinline version is faster (803.6 µs vs 874 µs). For MM at -O3, the timings are also very close (for example, at 2048×2048: 4.49874e+06 µs noinline vs 4.39603e+06 µs inline). 

Overall, the data suggests that the optimization level (-O0 vs -O3) has a much larger effect on performance than manually adding inline, and that inlining is most useful for very small helpers in hot loops, but its benefit must be confirmed empirically rather than assumed.



### 6. Performance Bottlenecks and Profiling

From our profiling results, the biggest bottleneck in the initial implementations was the naive matrix-matrix multiplication function, `multiply_mm_naive`. Time Profiler showed that most of the runtime was spent there, which meant the main issue was inside the actual computation kernel, not in setup or printing. The main reason it was slow was its memory access pattern: in the inner loop, matrix B was being accessed with poor locality because we were effectively moving down a column of a row-major matrix. That caused worse cache usage as the matrices got larger.

These profiling results helped us focus on optimizing memory access instead of small code-level overhead. Based on that, we used the transposed-B approach, where B is transposed first so the inner loop reads memory more contiguously. This improved cache locality and gave better performance in the benchmarks. So overall, profiling helped us identify that cache-unfriendly access to B was the main bottleneck and guided us toward an optimization that addressed it.

### 7. Teamwork and Collaboration

Splitting the work helped because each person could focus on a different part of the implementation. The main benefit was faster progress and cleaner separation of tasks. The main challenge was making sure all files used the same function signatures, memory layout, and benchmarking format.
