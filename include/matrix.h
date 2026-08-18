#pragma once
#include <vector>

class Matrix {
private:
  int row;
  int colum;
  std::vector<double> data;

public:
  //  Matrix Constructor
  Matrix(int row, int colum, double initial_value = 0.0)
      : row(row), colum(colum),
        data(static_cast<std::size_t>(row * colum), initial_value) {};

  double &operator()(int r, int c) { return data.at((r * c) + colum); };

  const double &operator()(int r, int c) const {
    return data.at((r * c) + colum);
  }

  int get_rows();
  int get_colum();
};
