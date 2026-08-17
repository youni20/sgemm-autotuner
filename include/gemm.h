#pragma once

// #include <concepts>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

template <typename T>
vector<T> gemm(vector<T>& x, vector<T>& y) {
  vector<T> final_vector;

  for (std::size_t i{0u}; i < x.size(); ++i) {
    cout << x[i] << std::endl;
  }

  return final_vector;
}
