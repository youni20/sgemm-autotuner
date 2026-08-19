#pragma once
#include <vector>

class Matrix {
private:
    int row;
    int colum;
    std::vector<float> data;

public:
    //  Matrix Constructor
    Matrix(int row, int colum, double initial_value = 0.0)
        : row(row)
        , colum(colum)
        , data(static_cast<std::size_t>(row * colum), initial_value) { };

    float& operator()(int r, int c) { return data.at((r * colum) + c); };

    const float& operator()(int r, int c) const
    {
        return data.at((r * colum) + c);
    };

    int get_rows() const;
    int get_colum() const;
    int get_size() const;

    void print_matrix();
    void randomize();
};
