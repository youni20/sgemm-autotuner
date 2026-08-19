#include <chrono>
#include <iostream>

#include "gemm.h"
#include "matrix.h"

int main()
{
    srand(time(NULL)); // Needed here for the randomize function
    int size { 1024 };

    Matrix A { size, size }; // Increased the matrix size to 1024x1024
    Matrix B { size, size }; // Since math works correct now we make it faster

    A.randomize();
    B.randomize();

    int repetitions { 0 };

    std::cout << "Execution Began...\n"
              << std::flush;

    while (repetitions != 3) { // Repeat a few times to get an avg time
        std::cout << "Execution " << repetitions + 1 << " Time: " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        Matrix C = gemm(A, B);

        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<std::chrono::microseconds>(end_time - start_time);

        ++repetitions;

        // GFLOP Calculation
        double seconds = duration.count() / 1e6;
        double total_flops = 2.0 * size * size * size;

        double giga_flops = total_flops / 1e9;
        double gflops_per_second = giga_flops / seconds;

        std::cout << duration.count()
                  << " microseconds | Performance: "
                  << gflops_per_second
                  << " GFLOP/s"
                  << std::endl;
    }

    std::cout << "Execution Complete" << std::endl;

    return 0;
}

/*
Matrix A{2, 3}; // 2x3
A(0, 0) = 1;
A(0, 1) = 2;
A(0, 2) = 3;
A(1, 0) = 4;
A(1, 1) = 5;
A(1, 2) = 6;

Matrix B(3, 2); // 3x2
B(0, 0) = 7;
B(0, 1) = 8;
B(1, 0) = 9;
B(1, 1) = 10;
B(2, 0) = 11;
B(2, 1) = 12;
*/
