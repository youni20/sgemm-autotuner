#include "gemm.h"
#include "matrix.h"
#include <vector>

using std::vector;

void print_answer(const vector<vector<int>> &x) {
  for (vector<int> row : x) {
    std::cout << "{ ";
    for (std::size_t j{0u}; j < x.size(); ++j) {
      std::cout << row[j] << " ";
    }
    std::cout << "}" << std::endl;
  }
}

int main() {
  Matrix A{2, 3};
  A(0,0) = 1;
  A(0,1) = 2;
  A(0,2) = 3;
  A(1,0) = 4;
  A(1,1) = 5;
  A(1,2) = 6;


  return 0;
}

/*
vector<vector<int>> matrix_a = {// 2x3
                                {1, 2, 3},
                                {4, 5, 6}};

vector<vector<int>> matrix_b = {// 3x2
                                {7, 8},
                                {9, 10},
                                {11, 12}};

// std::cout << matrix_a.at(1).at(2) << std::endl;
// std::cout << matrix_a.size() << std::endl;

*/
