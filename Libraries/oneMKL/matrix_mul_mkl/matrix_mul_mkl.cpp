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
static
bool test(queue &Q, int M, int N, int K)
{
    std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") matrix multiplication, " << type_string<T>() << "\n";;

    std::cout << " -> Initializing data...\n";

    /* Allocate A/B/C matrices */
    int lda = nice_ld<T>(M);
    int ldb = nice_ld<T>(K);
    int ldc = nice_ld<T>(M);

    auto A = malloc_device<T>(lda * K, Q);
    auto B = malloc_device<T>(ldb * N, Q);
    auto C = malloc_device<T>(ldc * N, Q);

    constexpr int rd_size = 1048576;
    std::vector<T> host_vector(rd_size);
    auto host_data = host_vector.data();

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

    /* Fill A/B with all ones to verify correctness */
    generate_ones(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_data, rd_size);
    replicate_data(Q, B, ldb * N, host_data, rd_size);

    /* Verify that the leading entries of C are correct */
    std::cout << " -> Verification...";
    (void) time_gemms(1);
    size_t elems = std::min(ldc * N, rd_size);
    Q.copy(C, host_data, elems).wait();
    bool ok = true;
    int linear_id = 0;
    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < M; i++) {
            linear_id = j*ldc + i;
            if (linear_id >= elems) break;
            if (host_data[linear_id] != T(K)) {
                ok = false;
            }
        }
        if (linear_id >= elems) break;
    }

    std::cout << "gemm " << (ok ? " passes." : " FAILS!") << " for type: " << type_string<T>() << "\n";
    if (!ok) { return false; }

    /* Fill A/B with random data */
    generate_random_data(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_data, rd_size);
    replicate_data(Q, B, ldb * N, host_data, rd_size);

    /* Do a warmup call with random data to initialize MKL and ensure kernels are JIT'ed if needed */
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

    std::cout << "\nAverage performance: " << flops << unit << 'F' << "\n";

    /* Free data */
    free(C, Q);
    free(B, Q);
    free(A, Q);

    return true;
}

static
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
              << "   all (runs all above)\n"
              << "\n"
              << "This benchmark uses the default DPC++ device, which can be controlled using\n"
              << "  the ONEAPI_DEVICE_SELECTOR environment variable\n";
    std::exit(1);
}

static
bool device_has_fp64(sycl::device const& D) {
    return (D.get_info<sycl::info::device::double_fp_config>().size() != 0);
}

static
void device_info(sycl::device const& D) {
    std::cout << "oneMKL DPC++ GEMM benchmark\n"
              << "---------------------------\n"
              << "Platform:                " << D.get_platform().get_info<info::platform::name>()         << "\n"
              << "Device:                  " << D.get_info<info::device::name>()                          << "\n"
              << "Driver_version:          " << D.get_info<info::device::driver_version>()                << "\n"
              << "Core/EU count:           " << D.get_info<info::device::max_compute_units>()             << "\n"
              << "Maximum clock frequency: " << D.get_info<info::device::max_clock_frequency>() << " MHz" << "\n"
              << "FP64 capability:         " << (device_has_fp64(D) ? "yes" : "no") << "\n"
              << "\n"
              ;
}

int main(int argc, char **argv)
{
    auto pname = argv[0];
    int M = 4096, N = 4096, K = 4096;
    std::string type = "none";

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

    bool g_success = true;
    try { 
        device D(default_selector_v);
        device_info(D);

        context C(D);
        queue Q(C, D);

        if ("none" == type)
            std::string type = device_has_fp64(D) ? "double" : "float";

        if (type == "double") {
            if (device_has_fp64(D))
                test<double>(Q, M, N, K);
            else {
                std::cout << "no FP64 capability on given SYCL device and type == \"double\"";
                return 1;
            }
        }
        else if (type == "single" || type == "float")
            g_success = g_success && test<float>(Q, M, N, K);
        else if (type == "half")
            g_success = g_success && test<half>(Q, M, N, K);
        else if (type == "all") {
            type = "half";
            g_success = g_success && test<half>(Q, M, N, K);

            type = "float";
            g_success = g_success && test<float>(Q, M, N, K);

            if (device_has_fp64(D)) {
                type = "double";
                g_success = g_success && test<double>(Q, M, N, K);
            }
        } else {
            type = "none";
            usage(pname);
        }
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        std::cerr << " while performing GEMM for" 
            << " M=" << M 
            << ", N=" << N
            << ", K=" << K
            << ", type `" << type << "`"
            << "\n";
        return 139;
    }

    return g_success ? 0 : 1;
}

