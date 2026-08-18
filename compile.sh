#!/bin/bash

clang-format -i -style=webkit include/gemm.h include/matrix.h src/main.cc src/matrix.cc src/gemm.cc
clang++ @compile_flags.txt src/main.cc src/matrix.cc src/gemm.cc -o build/app
