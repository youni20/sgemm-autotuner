#pragma once

#include <iostream>
#include <vector>
#include <concepts>

using std::vector;


template <typename T>
requires std::integral<T> || std::floating_point<T>
vector<T> gemm(vector<T>& x, vector<T>& y){
    vector<T> final_vector;

    for(std::size_t column_value : x){
        std::cout << column_value;
    }

    return final_vector;
}
