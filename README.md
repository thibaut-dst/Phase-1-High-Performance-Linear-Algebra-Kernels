# Phase-1-High-Performance-Linear-Algebra-Kernels



Repo structure

root/
├─ .gitignore
├─ README.md              # team, build, how to run, answers to discussion Qs
├─ report.pdf             # the formal write‑up
├─ include/
│   └─ linalg.hpp         # function declarations for all kernels
├─ src/
│   ├─ linalg.cpp         # implementations of all kernels (Part 1 + optimized)
│   ├─ main_baseline.cpp  # simple correctness tests / basic runs
│   └─ main_bench.cpp     # all benchmarking / profiling entry point
└─ CMakeLists.txt or simple build instructions in README