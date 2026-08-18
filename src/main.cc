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
  Matrix A{2, 3};  // 2x3
  A(0,0) = 1;
  A(0,1) = 2;
  A(0,2) = 3;
  A(1,0) = 4;
  A(1,1) = 5;
  A(1,2) = 6;

  Matrix B(3,2);  // 3x2
  B(0,0) = 7;
  B(0,1) = 8;
  B(1,0) = 9;
  B(1,1) = 10;
  B(2,0) = 11;
  B(2,1) = 12;

  return 0;
}
