#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
// #include <concepts>

using std::cout;
using std::vector;

template <typename T>
vector<vector<T>> gemm(vector<T> &matrix_a, vector<T> &matrix_b) {
  if ((matrix_a[0].size() != matrix_b.size())) {
    throw std::invalid_argument(
        "ERROR: Matrices are not compatible for multiplication\n");
  } // Check if no. of colums of matrix A = no. of rows in matrix B

  vector<vector<T>> final_matrix;

  for (std::size_t i{0u}; i < matrix_a.size(); ++i) { // number of rows in A
    for (std::size_t j{0u}; j < matrix_b[0].size();
         ++j) { // number of columns in B
      int running_total{0};
      for (std::size_t k{0u}; k < matrix_a[0].size(); ++k) {
        running_total += matrix_a[i][k] * matrix_b[k][j];
      }
      final_matrix[i][j] = running_total;
    }
  }

  return final_matrix;
}
