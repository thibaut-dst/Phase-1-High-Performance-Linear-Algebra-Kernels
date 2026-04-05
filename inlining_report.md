# Inlining Analysis Summary

This document examines how the `inline` keyword and compiler optimization level interact to affect the performance of matrix-vector and matrix-matrix multiplication kernels, and when explicit inlining provides a genuine speedup versus when the compiler renders it unnecessary.

## Background

Every function call in C++ carries overhead that is invisible in the source code but real in the generated machine code. When a function is called, the CPU must save the current instruction pointer, push arguments into registers or onto the stack, jump to the function body, execute it, then restore the saved state and return. For a large function this cost is negligible relative to the work performed. For a small helper — such as a single fused-multiply-add operation — called millions of times inside a matrix inner loop, the call machinery can represent a substantial fraction of total runtime.

The `inline` keyword instructs the compiler to substitute the function body directly at the call site rather than emitting a call instruction. This eliminates the jump and register overhead entirely, at the cost of slightly larger binary size if the function is called from many locations.

## Experiment Design

A small helper function `fma_op` (computing `acc + a * b`) was defined in two versions — one plain, one marked `inline` — and substituted into otherwise identical row-major matrix-vector and transposed-B matrix-matrix kernels. Both kernel pairs were benchmarked at two compiler optimization levels:

- **`-O0`**: no optimization; the compiler emits straightforward machine code, respecting every call boundary.
- **`-O3`**: aggressive optimization; the compiler applies loop unrolling, auto-vectorisation, and its own inlining heuristics.

All matrix sizes (128×128 through 2048×2048) used identical random input data, isolating inlining as the only variable within each optimization tier.

## Results at -O0

At no optimization, the non-inline variants were consistently slower across all matrix sizes. Each iteration of the inner loop emitted a real `CALL` instruction to `fma_op`. For a 2048×2048 matrix-vector multiply the inner loop executes approximately 4 million times, producing 4 million function calls per benchmark run. The inline variants eliminated this overhead entirely, yielding a measurable and stable speedup.

The effect was most visible in the matrix-vector kernels. The MV inner loop is short — each iteration does one multiply and one addition — so the call overhead represents a large share of the per-iteration cost. In the matrix-matrix kernels the per-iteration arithmetic is identical, but the triple-nested loop structure and larger absolute runtimes made the percentage improvement appear smaller even though the raw overhead was the same.

## Results at -O3

At aggressive optimization, the gap between the inline and non-inline variants largely disappeared. The compiler's own inliner analyses call sites and inlines small functions automatically, regardless of whether the `inline` keyword is present. Both `fma_op` and `fma_op_inline` were inlined by the compiler in practice, so the explicit keyword provided no additional benefit.

Both variants were also substantially faster than their `-O0` counterparts. At `-O3` the compiler additionally applied loop unrolling and SIMD vectorisation to the inner loops, transforming what had been scalar instructions into wide vector operations processing multiple elements per cycle. This effect was far larger than any inlining gain, demonstrating that optimization level has a greater overall impact on performance than the `inline` keyword alone.

## Trade-offs and When Inlining Helps

Inlining is most beneficial when three conditions hold simultaneously: the helper function is small (few instructions), it is called inside a hot inner loop (high call frequency), and the compiler optimization level is low or moderate. Under these conditions, eliminating the call overhead produces a consistent improvement.

The principal downside of inlining is code bloat. If a function is inlined at many distinct call sites, the compiler emits a separate copy of its body at each location. For small helpers like `fma_op` this is negligible, but for larger functions it increases binary size and can evict useful instructions from the instruction cache — a phenomenon known as I-cache pressure, which can degrade performance in programs with many distinct hot paths.

## Conclusion

The `inline` keyword is a reliable, low-risk optimisation for small helper functions called in tight loops, particularly when compiling at `-O0` or `-O1`. At `-O3` the compiler makes these decisions autonomously and the explicit hint becomes advisory rather than directive. The larger lesson from this experiment is that compiler optimisation level dominates: the jump from `-O0` to `-O3` produced improvements orders of magnitude larger than the inline keyword alone, driven primarily by auto-vectorisation and loop transformations rather than call elimination.
