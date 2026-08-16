# Single-Precision Matrix Multiplication

This project is an implementation of a Single-Precision matrix multiplication in C++, taking it from a simple triple-loop implementation (with a time complexity of O(n^3)) to a version that approaches a more practical peak performance off the CPU.

Once thats complete an autotuner was then implemented that searches the optimisation parameter space based of data rather than by hand-picked constants. The goal is to benchmark it against OpenBLAS as an external reference at every stage. I also aim to give the reasoning behind each optimisation and not just the resulting numbers, as this aims to be a high performance computing project.
