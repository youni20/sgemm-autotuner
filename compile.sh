#!/bin/bash

clang-format -i -style=llvm include/gemm.h include/matrix.h src/main.cc src/matrix.cc
clang++ @compile_flags.txt src/main.cc src/matrix.cc -o build/app
