#include <vector>
#include "gemm.h"

int main(){
    std::vector<int> x = {2,4,6,8};
    std::vector<int> y = {3,6,9,12};

    gemm(x, y);
    return 0;
}
