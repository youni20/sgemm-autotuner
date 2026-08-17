#include "gemm.h"
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
  vector<vector<int>> matrix_a = {// 2x3
                                  {1, 2, 3},
                                  {4, 5, 6}};

  vector<vector<int>> matrix_b = {// 3x2
                                  {7, 8},
                                  {9, 10},
                                  {11, 12}};

  // std::cout << matrix_a.at(1).at(2) << std::endl;
  // std::cout << matrix_a.size() << std::endl;

  vector<vector<int>> matrix_c = gemm(matrix_a, matrix_b);
  print_answer(matrix_c);

  return 0;
}
