#pragma once
#include <iostream>
#include <vector>
using std::vector;


template <typename T> vector<T> gemm(vector<T>& x, vector<T>& y){
    vector<T> final_vector;

    for(std::size_t column_value : x){
        std::cout << column_value;
    }

    return final_vector;
}
