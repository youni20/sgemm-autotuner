#include "matrix.h"

#include <cstdlib>
#include <iostream>

int Matrix::get_rows() const { return row; };
int Matrix::get_colum() const { return colum; };
int Matrix::get_size() const { return data.size(); };

void Matrix::print_matrix()
{
    for (int row { 0u }; row < this->get_rows(); ++row) {
        std::cout << "{ ";
        for (int colum { 0u }; colum < this->get_colum(); ++colum) {
            std::cout << this->data[row * this->get_colum() + colum] << " ";
        }
        std::cout << "}" << std::endl;
    }
}

void Matrix::randomize()
{
    // srand(time(NULL));
    for (std::size_t i { 0u }; i < data.size(); ++i) {
        int x = (rand() % 100) + 1; // Generate a random number between 1 - 100
        data[i] = x; // Place the random number to the vector
    }
}
