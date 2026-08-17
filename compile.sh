#!/bin/bash

clang-format -i -style=llvm include/gemm.h src/main.cc
clang++ @compile_flags.txt src/main.cc -o build/app
