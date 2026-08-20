#pragma once
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64 // Allow the compiler to define the block size and if it doesnt default to 64
#endif

#include "matrix.h"

Matrix gemm(const Matrix& matrix_a, const Matrix& matrix_b);
