#pragma once

// #include <concepts>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

template <typename T> vector<T> gemm(vector<T> &x, vector<T> &y) {
  vector<T> final_vector;

  for (vector i : x) {
    for (std::size_t j : i) {
      std::cout << j << std::endl;
    };
  }

  return final_vector;
}
