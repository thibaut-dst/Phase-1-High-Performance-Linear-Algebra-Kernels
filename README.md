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



Test for part 1, run:
g++ -O0 src/main_baseline.cpp src/linalg.cpp -Iinclude -o baseline
./baseline

Test for part 2.2 and 2.3, run:
g++ -O3 src/main_bench.cpp src/linalg.cpp -Iinclude -o bench
./bench > benchmarks.csv