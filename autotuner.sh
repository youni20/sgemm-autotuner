#!/bin/bash

mkdir -p build

# The search space: We will test block sizes from 16x16 up to 256x256
BLOCK_SIZES=(16 32 64 128 256)

echo "============================================="
echo " Starting GEMM Autotuner..."
echo " Matrix Size: 1024x1024"
echo " Search Space: ${BLOCK_SIZES[*]}"
echo "============================================="

# Iterate through every block size in our array
for size in "${BLOCK_SIZES[@]}"; do
    echo "--> Compiling and testing BLOCK_SIZE = $size..."

    # Compile the code, injecting the -DBLOCK_SIZE flag
    clang++ -xc++ -std=c++20 -Wall -Wextra -O3 -march=native -mavx2 -mfma -fopenmp \
            -DBLOCK_SIZE=$size \
            -Iinclude src/main.cc src/matrix.cc src/gemm.cc -o build/autotuner

    # Run the executable, but only print the lines containing "Performance"
    ./build/autotuner | grep "Performance"

    echo "---------------------------------------------"
done

echo "Autotuning Complete!"
