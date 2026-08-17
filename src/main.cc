#include "gemm.h"
#include <vector>

using std::vector;

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

  return 0;
}
