//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Contents:
//     A simple matrix multiplication benchmark, using the oneAPI Math Kernel
//     Library (oneMKL).
//

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "utilities.hpp"

using namespace sycl;

template <typename T>
void test(queue &Q, int M, int N, int K)
{
    std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") matrix multiplication, " << type_string<T>() << std::endl;;

    std::cout << " -> Initializing data...\n";

    /* Allocate A/B/C matrices */
    int lda = nice_ld<T>(M);
    int ldb = nice_ld<T>(K);
    int ldc = nice_ld<T>(M);

    auto A = malloc_device<T>(lda * K, Q);
    auto B = malloc_device<T>(ldb * N, Q);
    auto C = malloc_device<T>(ldc * N, Q);

    /* Fill A/B with random data */
    constexpr int rd_size = 1048576;
    auto random_data = malloc_host<T>(rd_size, Q);
    generate_random_data(rd_size, random_data);

    replicate_data(Q, A, lda * K, random_data, rd_size);
    replicate_data(Q, B, ldb * N, random_data, rd_size);

    /* Measure time for a given number of GEMM calls */
    auto time_gemms = [=, &Q](int runs) -> double {
        using namespace oneapi::mkl;
        using namespace std::chrono;
        auto start = steady_clock::now();
        for (int i = 0; i < runs; i++)
            blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
        Q.wait_and_throw();
        auto end = steady_clock::now();
        return duration<double>(end - start).count();
    };

    /* Do a warmup call to initialize MKL and ensure kernels are JIT'ed if needed */
    std::cout << " -> Warmup...\n";
    (void) time_gemms(1);

    /* Time one GEMM call, and estimate how many calls will be required to keep the
     * GPU busy for 1s. */
    auto tare = time_gemms(1);
    int ncalls = std::max(4, std::min(1000, int(1. / tare)));

    /* Time that many GEMMs, subtracting the first call time to remove host overhead.
     * This gives a better idea of device performance. */
    std::cout << " -> Timing...\n";
    auto time = time_gemms(ncalls + 1) - tare;
    auto avg = time / ncalls;

    /* Calculate and display performance */
    auto op_count = double(M) * double(N) * double(K) * 2;
    auto flops = op_count / avg;

    flops *= 1e-9;
    char unit = 'G';
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'T';
    }
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'P';
    }

    std::cout << "\nAverage performance: " << flops << unit << 'F' << std::endl;

    /* Free data */
    free(A, Q);
    free(B, Q);
    free(C, Q);
    free(random_data, Q);
}

void usage(const char *pname)
{
    std::cerr << "Usage:\n"
              << "  " << pname << " [type] N           benchmark (NxN) x (NxN) square matrix multiplication (default: N = 4096)\n"
              << "  " << pname << " [type] M N K       benchmark (MxK) x (KxN) square matrix multiplication\n"
              << "\n"
              << "The optional [type] selects the data type:\n"
              << "   double    [default]\n"
              << "   single\n"
              << "   half\n"
              << "\n"
              << "This benchmark uses the default DPC++ device, which can be controlled using\n"
              << "  the ONEAPI_DEVICE_SELECTOR environment variable\n";
    std::exit(1);
}

int main(int argc, char **argv)
{
    auto pname = argv[0];
    int M = 4096, N = 4096, K = 4096;
    std::string type = "double";

    if (argc <= 1)
        usage(pname);

    if (argc > 1 && std::isalpha(argv[1][0])) {
        type = argv[1];
        argc--; argv++;
    }

    if (argc > 1) M = N = K = std::atoi(argv[1]);

    if (argc > 3) {
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    if (M <= 0 || N <= 0 || K <= 0)
        usage(pname);

    queue Q;

    std::cout << "oneMKL DPC++ GEMM benchmark\n"
              << "---------------------------\n"
              << "Device:                  " << Q.get_device().get_info<info::device::name>()                          << std::endl
              << "Core/EU count:           " << Q.get_device().get_info<info::device::max_compute_units>()             << std::endl
              << "Maximum clock frequency: " << Q.get_device().get_info<info::device::max_clock_frequency>() << " MHz" << std::endl;

    if (type == "double")
        test<double>(Q, M, N, K);
    else if (type == "single" || type == "float")
        test<float>(Q, M, N, K);
    else if (type == "half")
        test<half>(Q, M, N, K);
    else
        usage(pname);
}
