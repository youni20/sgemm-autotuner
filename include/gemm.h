#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
// #include <concepts>

using std::cout;
using std::vector;

template <typename T> vector<T> gemm(vector<T> &x, vector<T> &y) {
  if ((x[0].size() != y.size())) { // Check if no. of colums of matrix A = no.
                                   // of rows in matrix B
    throw std::invalid_argument("ERROR: Matrices are not compatible for multiplication\n");
  }

  vector<T> final_vector;

  for (vector i : x) {
    for (std::size_t j : i) {
      std::cout << j << std::endl;
    };
  }

  return final_vector;
}
