# Single-Precision Matrix Multiplication

This project is an implementation of a Single-Precision matrix multiplication in C++, taking it from a simple triple-loop implementation (with a time complexity of O(n^3)) to a version that approaches a more practical peak performance off the CPU.

Once thats complete an autotuner was then implemented that searches the optimisation parameter space based of data rather than by hand-picked constants. The goal is to benchmark it against OpenBLAS as an external reference at every stage. I also aim to give the reasoning behind each optimisation and not just the resulting numbers, as this aims to be a high performance computing project. I also document my findings and progress throughout as a implementation.

Lastly this is an opensource project under the MIT LICENSE so any contributions and further work is more than welcome!

---

## First Implementation:

Here is my first implementation of the gemm process which works but obviously isnt optimised at the moment.
<br>
![First Prototype](images/implementation1.png)
<br>

Not only is this implementation a time complexity of O(n^3) which makes it inefficient but it also uses a vector within a vector. My initial thoughts were considering the fact this is matrix multiplication im attempting to implement it would only make sense for the factors being multiplied to be 2D matrices however upon further research I found that this is terrible for the cpu for various reasons.

1. With "vector<vector<int>>" the issue is that the outer vector just holds memory addresses, and each of these addresses points to an inner vector representing a row, which is located somewhere else within the heap memory.

2. Because of this when the cpu attempts to perform the matrix multiplication it has to constatly look around to different memory locations instead of a contiguos location to find the next row. This is known as pointer chasing. This causes cache misses which cause the CPU to stall while it waits for data from RAM destroying the performance.

3. To solve this the cpu needs all the matrix data laid out in one single continuos block of memory, allowing the hardware to predict what data will be needed next (prefetching) and process multiple numbers at the same time (SIMD Vectorization).

#### Row Major Trick
The solution is to flatten the grid, since the computers ram is not like a grid but rather a long continuos line of bytes. Row 0 comes first, imediantly followed by Row 1 and then Row 2. To trick the C++ code into treating this like a 1D line like a 2D grid we use a simple math formula to calculate exactly where a (row, col) coordinate lands in the 1D line:

**index = (row * TOTAL_COLUMNS) + col**


## Implementation 2:

![Moved to a 1D matrix to "trick" the cpu in a way](images/implementation2.png)

Now it is better for the cpu as the data is stored contiguosly instead of scattered accross the heap. I have now implemented the baseline single-precision matrix multiplication (gemm). 

For a matrix size of 1024x1024, the unoptimized triple-loop prototype running on the contiguous 1D layout completed in around 5x10^6 microseconds accross multiple runs.

While storing the data contiguously avoids pointer chasing and cache misses compared to a std::vector<std::vector<float>>, this naive O(n^3) implementation still suffers from poor cache reuse and lacks vectorization. Future iterations will introduce tiling, SIMD, and multithreading to drastically improve performance.

![Result of running the program at this stage on 2 1024x1024 matrices](images/implementation3.png)
