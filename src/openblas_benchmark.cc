#include <cblas.h> // The OpenBLAS header
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    int size = 1024;

    // Allocate contiguous 1D memory for the matrices
    std::vector<float> A(size * size);
    std::vector<float> B(size * size);
    std::vector<float> C(size * size, 0.0f);

    // Randomize the matrices
    srand(time(NULL));
    for (int i = 0; i < size * size; ++i) {
        A[i] = static_cast<float>(rand() % 100);
        B[i] = static_cast<float>(rand() % 100);
    }

    std::cout << "OpenBLAS Execution Began..." << std::endl;

    // Run 3 iterations to match my custom benchmark format
    for (int run = 1; run <= 3; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // The official SGEMM function
        // SGEMM = Single-Precision General Matrix Multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            size, size, size, // M, N, K dimensions
            1.0f, // Alpha (multiplier for A*B)
            A.data(), size, // Matrix A and its row stride
            B.data(), size, // Matrix B and its row stride
            0.0f, // Beta (multiplier for C)
            C.data(), size); // Matrix C and its row stride

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Calculate GFLOP/s
        double seconds = duration.count() / 1e6;
        double total_flops = 2.0 * size * size * size;
        double gflops_per_sec = (total_flops / 1e9) / seconds;

        std::cout << "Execution " << run << " Time: " << duration.count()
                  << " microseconds | Performance: " << gflops_per_sec << " GFLOP/s\n";
    }

    std::cout << "Execution Complete" << std::endl;
    return 0;
}
