#include "gemm.h"
#include <algorithm>
#include <stdexcept>
// #include <concepts>

Matrix gemm(const Matrix& matrix_a, const Matrix& matrix_b)
{
    if (matrix_a.get_colum() != matrix_b.get_rows()) {
        throw std::invalid_argument(
            "ERROR: Matrices are not compatible for multiplication\n");
    }

    // Initialize the output matrix with zeros
    Matrix final_matrix(matrix_a.get_rows(), matrix_b.get_colum(), 0.0f);

    // Set the block size so we can now counter Cache Blocking
    const int BLOCK_SIZE { 64 };

    // Instead of stepping by 1, these loops jump forward by BLOCK_SIZE
    for (int i_block { 0 }; i_block < matrix_a.get_rows(); i_block += BLOCK_SIZE) {
        for (int k_block { 0 }; k_block < matrix_a.get_colum(); k_block += BLOCK_SIZE) {
            for (int j_block { 0 }; j_block < matrix_b.get_colum(); j_block += BLOCK_SIZE) {

                // We use std::min to prevent going out of bounds if the matrix size
                // if not perfectly divisible by 64 (e.g., a 1000x1000 matrix).

                int i_end = std::min(i_block + BLOCK_SIZE, matrix_a.get_rows());
                int k_end = std::min(k_block + BLOCK_SIZE, matrix_a.get_colum());
                int j_end = std::min(j_block + BLOCK_SIZE, matrix_b.get_colum());

                // 1. Outer loop: Iterate through the rows of Matrix A
                for (int i = i_block; i < i_end; ++i) {

                    // 2. Middle loop: Iterate through the columns of Matrix A (and rows of B)
                    for (int k = k_block; k < k_end; ++k) {

                        // Cache this value because it doesn't change at all during the 'j' loop
                        float a_ik = matrix_a(i, k);

                        // 3. Inner loop: Iterate through the columns of Matrix B
                        for (int j = j_block; j < j_end; ++j) {

                            // Both final_matrix and matrix_b are now stepping forward
                            // exactly 1 memory address at a time. The hardware prefetcher will love this!
                            final_matrix(i, j) += a_ik * matrix_b(k, j);
                        }
                    }
                }
            }
        }
    }
    return final_matrix;
}

/*
Matrix gemm(const Matrix& matrix_a, const Matrix& matrix_b)
{
    if (matrix_a.get_colum() != matrix_b.get_rows()) {
        throw std::invalid_argument(
            "ERROR: Matrices are not compatible for multiplication\n");
    } // Check if no. of colums of matrix A = no. of rows in matrix B

    Matrix final_matrix(matrix_a.get_rows(), matrix_b.get_colum());

    for (int i { 0u }; i < matrix_a.get_rows(); ++i) { // number of rows in A
        for (int j { 0u }; j < matrix_b.get_colum(); ++j) { // number of columns in B
            float running_total { 0.0f };
            for (int k { 0u }; k < matrix_a.get_colum(); ++k) {
                running_total += matrix_a(i, k) * matrix_b(k, j);
            }
            final_matrix(i, j) = running_total;
        }
    }

    return final_matrix;
};
*/
