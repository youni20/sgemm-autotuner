# Single-Precision GEMM in C++

An implementation of single-precision general matrix multiplication (GEMM) in C++, progressing from a naive triple-loop baseline towards a version that approaches practical peak CPU performance. Once the hand-optimised path is complete, an autotuner will search the optimisation parameter space empirically rather than relying on hand-picked constants. Every stage is benchmarked against OpenBLAS as an external reference, with the reasoning behind each optimisation documented alongside the resulting numbers.

This is a high-performance computing project in the literal sense: the goal is not just a working GEMM, but a record of *why* each transformation improves performance, backed by measurement.

Open source under the MIT licence. Contributions and further work are welcome.

---

## Roadmap

- [x] Naive triple-loop baseline (`vector<vector<T>>`)
- [x] Flatten to contiguous 1D row-major storage
- [x] Cache blocking / tiling
- [x] SIMD vectorisation (AVX2/AVX-512)
- [x] Multithreading
- [ ] Autotuner for block size and tiling parameters
- [ ] Benchmark against OpenBLAS at every stage

---

## Implementation Log

### v1: Naive Triple-Loop, `vector<vector<T>>`

![First prototype](images/implementation1.png)

The first working implementation, with the expected $O(n^3)$ time complexity. Matrices are represented as `vector<vector<T>>`, which seemed like the natural choice for a 2D mathematical object but turns out to be a poor fit for CPU performance, for a few reasons:

1. **No contiguity.** A `vector<vector<T>>` is an outer vector of pointers, where each pointer refers to an inner vector (a row) allocated independently somewhere on the heap.
2. **Pointer chasing.** To reach a given row, the CPU must first dereference the outer vector, then follow that pointer to a separate heap allocation. Rows are not guaranteed to be adjacent in memory, or even nearby.
3. **Cache misses.** This scattered layout defeats the CPU's prefetcher and produces cache misses on nearly every row access, stalling the pipeline while data is fetched from RAM.

The fix is to give the CPU one contiguous block of memory to work with, which enables hardware prefetching and makes SIMD vectorisation possible in later stages.

### v2: Flattened to 1D, Row-Major Order

![Flattened to a 1D contiguous buffer](images/implementation2.png)

RAM is not a grid, it is a single contiguous line of bytes. The matrix is flattened accordingly: row 0 is followed immediately by row 1, then row 2, and so on. A `(row, col)` coordinate is mapped onto this 1D buffer with the standard row-major indexing formula:

$$
\text{index} = (\text{row} \times \text{total\_columns}) + \text{col}
$$

This keeps the entire matrix in one contiguous allocation, removing the pointer chasing from v1.

---

## Benchmark Methodology

Wall-clock execution time alone is a poor performance metric: it is sensitive to CPU clock fluctuations, thermal throttling, and hardware differences across machines. To get a hardware-agnostic figure, throughput is reported in **GFLOP/s** (giga floating-point operations per second) instead.

For an $N \times N \times N$ matrix multiplication, the total floating-point operation count is:

$$
\text{FLOPs} = 2N^3
$$

For $N = 1024$, this is approximately 2.15 billion floating-point operations per run.

## Results

![Three benchmark runs on identical input matrices](images/implementation3.png)

Baseline (v2, naive loop order, contiguous 1D storage), compiled with `-O3 -march=native`, 1024×1024 matrices, three runs:

| Run | Time (µs) | Throughput (GFLOP/s) |
|-----|-----------|-----------------------|
| 1   | 3,515,659 | 0.611 |
| 2   | 3,482,995 | 0.617 |
| 3   | 3,497,846 | 0.614 |

**Average: ~3.50 s, ~0.61 GFLOP/s.**

### Analysis: the memory wall

A single modern CPU core is capable of tens to hundreds of GFLOP/s. Achieving only 0.61 GFLOP/s here is not a fluke, it is a direct consequence of memory access pattern rather than raw compute capability.

Contiguous storage alone removes pointer chasing but does not fix the underlying issue: the naive `i, j, k` loop order accesses matrix B column-wise in the innermost loop, striding through memory with poor spatial locality. This evicts useful data from L1/L2 cache long before it can be reused, so the compute units sit idle waiting on data from main memory far more often than they perform useful work. This is a **memory-bound**, not compute-bound, workload, and it demonstrates that compiler flags alone (`-O3`, `-march=native`) cannot fix an algorithm whose loop structure is fighting the cache hierarchy. The next stages (blocking/tiling, then vectorisation) address this directly.


### v3: True SGEMM + Loop Reordering (i, k, j)

![Post loop-permutation and float conversion results](images/implementation4.png)

Two changes were applied together in this step:

1. **Genuine single precision.** Every `double` in `Matrix` was replaced with `float`: the storage vector, the constructor's `initial_value`, and both `operator()` overloads. A 32-bit float doubles the number of elements that fit in a cache line and in a SIMD register relative to `double`, so this is a prerequisite for effective vectorisation later, independent of any loop reordering.

2. **Loop permutation, `i, j, k` → `i, k, j`.** In the original `i, j, k` order, the innermost loop walks `matrix_b(k, j)` along `k`, i.e. down a column, striding by `total_columns` elements on every iteration. For a 1024-wide matrix that is a 4096-byte stride, one new cache line touched per iteration, no reuse. Reordering so `j` is innermost makes the access `matrix_b(k, j)` walk along a row, i.e. unit stride. The hardware prefetcher recognises the sequential pattern and streams the row into L1 ahead of demand. `matrix_a(i, k)` is loop-invariant with respect to `j`, so it is hoisted out of the innermost loop and kept in a register (`a_ik`) rather than reloaded 1024 times.

```cpp
Matrix final_matrix(matrix_a.get_rows(), matrix_b.get_colum(), 0.0f);

for (int i = 0; i < matrix_a.get_rows(); ++i) {
  for (int k = 0; k < matrix_a.get_colum(); ++k) {
    float a_ik = matrix_a(i, k);          // loop-invariant, hoisted into a register
    for (int j = 0; j < matrix_b.get_colum(); ++j) {
      final_matrix(i, j) += a_ik * matrix_b(k, j);   // unit-stride access
    }
  }
}
```

Note the accumulation pattern also changed: v2 computed a full dot product into a scalar `running_total` before a single write to `final_matrix[i][j]`. v3 accumulates directly into `final_matrix(i, j)` across the `k` loop, trading one register-resident accumulator for repeated read-modify-write on the output matrix. At this matrix size the output row stays cache-resident across the `k` iterations, so the trade is favourable here, but it is a second, unmeasured variable stacked on top of the two above.

#### Results

1024×1024, `-O3 -march=native`, three runs:

| Run | Time (µs) | Throughput (GFLOP/s) |
|-----|-----------|-----------------------|
| 1   | 766,784   | 2.801 |
| 2   | 757,305   | 2.836 |
| 3   | 757,153   | 2.836 |

**Average: ~0.760 s, ~2.82 GFLOP/s**, a **4.6× improvement** over the v2 baseline (0.61 GFLOP/s).

#### Caveat: confounded variables

This result cannot be decomposed into "how much did `float` contribute" versus "how much did loop reordering contribute." The two changes were shipped together. Rigorous attribution requires an intermediate measurement:

- [x] Benchmark `float` storage with the **original** `i, j, k` loop order, isolating the type change
- [x] Benchmark `double` storage with the **new** `i, k, j` loop order, isolating the reordering

Without this ablation, "4.6×" is a bundled number, not evidence for either optimisation on its own. Given that `i, k, j` eliminates a column-strided access entirely while the `float`-vs-`double` change only affects cache line and register packing density, the loop reordering is almost certainly the dominant term, but "almost certainly" is not a measurement. Isolate before the next stage adds a third variable (SIMD) on top of two already-unattributed ones.

### v4: Cache Blocking (Tiling)

Loop reordering (v3) fixed spatial locality in the innermost loop but left temporal locality unaddressed at the level of the full matrix. For $N = 1024$, a row of `matrix_b` is $1024 \times 4\ \text{bytes} = 4\ \text{KiB}$, and the working set touched across one full sweep of the `k` loop exceeds L1 capacity well before the sweep completes. Data brought into L1 early in the `k, j` traversal is evicted before `i` advances far enough to reuse it, so the CPU is still making repeated round trips to L2/L3 for data it has already seen.

The fix is to restructure the computation into six nested loops instead of three: three outer loops iterate over block coordinates, and three inner loops perform the standard `i, k, j` multiplication strictly within the bounds of the current block. Matrices A, B and the accumulator are partitioned into $64 \times 64$ sub-blocks, sized so that the active tile of each operand fits in L1 simultaneously. Every multiply-accumulate for a given block is exhausted before the block is evicted, so each cache line loaded is reused across the full block rather than once.

#### Results

![1024×1024, -O3 -march=native, block size 64×64](images/implementation4.png)

This is a **~3.4× improvement** over v3 (2.82 GFLOP/s) and a **~15.6× improvement** over the v2 baseline (0.61 GFLOP/s).

#### Caveat: block size was not swept

64×64 was chosen by hand, not derived from the target machine's actual L1 capacity or associativity. The correct block size is a function of cache line size, L1 capacity, and how many of A's block, B's block and the accumulator's block must be simultaneously resident, none of which was measured here. Treat 64 as a placeholder pending the autotuner stage on the roadmap, which should sweep block size empirically against measured throughput rather than have it fixed by assumption.


### v5: SIMD Vectorisation (AVX2 + FMA)

Cache blocking (v4) resolved the memory-bound behaviour of the workload, shifting the bottleneck to the compute units themselves. Scalar `+` and `*` map to one float operation per instruction, so even with data resident in L1 the core is executing far below its theoretical throughput.

**The fix.** Two changes were made together:

1. **32-byte aligned storage.** `Matrix` now allocates through a custom `AlignedAllocator` backed by `std::aligned_alloc`, guaranteeing 32-byte alignment. This is a hard requirement for `_mm256_load_ps`/`_mm256_store_ps`; unaligned pointers passed to the aligned load/store variants fault rather than degrade gracefully.
2. **Manual AVX2/FMA intrinsics in the innermost loop**, replacing the scalar `i, k, j` body:

   - `_mm256_set1_ps` broadcasts `matrix_a(i, k)` into all eight lanes of a 256-bit register.
   - `_mm256_load_ps` reads eight contiguous `matrix_b` floats per iteration instead of one.
   - `_mm256_fmadd_ps` performs eight multiply-accumulates in a single instruction.
   - `_mm256_store_ps` writes the eight-wide result back.

This replaces eight scalar iterations of the innermost `j` loop with one vector iteration, within each 64×64 block established in v4.

#### Results

![1024×1024, `-O3 -march=native`, block size 64×64, three runs](images/implementation6.png)

**Average: ~0.213 s, ~10.08 GFLOP/s**, a **~1.06× improvement** over v4 (9.50 GFLOP/s).

#### Caveat: the gain is far below the 8× the SIMD width implies

An 8-wide FMA unit operating on data already resident in L1 should not yield a ~6% improvement. Candidate explanations, none confirmed:

- [ ] `-march=native` under `-O3` may already have auto-vectorised the v4 scalar loop, making the manual intrinsics redundant rather than additive. Compare against the v4 disassembly (`objdump -d` / Compiler Explorer) to check for existing `vfmadd` instructions before attributing any of this gain to the intrinsics.
- [ ] The 64×64 block size was tuned (informally, in v4) around scalar access patterns, not around register-blocking for 8-wide FMA. The block dimensions may now be suboptimal for the vector width and need re-deriving jointly, not left as an inherited constant.
- [ ] Store-side overhead: `_mm256_store_ps` on the accumulator every inner iteration re-introduces the same read-modify-write traffic on `final_matrix` flagged as an unmeasured variable back in v3, now happening eight floats at a time instead of one.

Profile with Nsight/`perf` before the next stage (multithreading) is layered on top; a compute-bound explanation should show near-zero L1 miss rate and high port utilisation on the FMA unit, and if it doesn't, vectorisation is not actually the limiting factor here.

---

## License

MIT. See `LICENSE`.
