#!/bin/bash

echo "Compiling..."

clang-format -i -style=webkit include/gemm.h include/matrix.h src/main.cc src/matrix.cc src/gemm.cc src/openblas_benchmark.cc
clang++ @compile_flags.txt src/main.cc src/matrix.cc src/gemm.cc -o build/app
clang++ -xc++ -std=c++20 -Wall -Wextra -O3 -march=native src/openblas_benchmark.cc -lopenblas -o build/openblas_benchmark

echo "Done"
