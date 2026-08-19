#include "gemm.h"
#include <stdexcept>
// #include <concepts>

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
