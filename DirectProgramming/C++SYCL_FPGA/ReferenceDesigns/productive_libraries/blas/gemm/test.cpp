#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include "mkl_cblas.h"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "./api.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

sycl::device d{sycl::cpu_selector_v};
std::vector<sycl::device*> devices{&d};

namespace {

template <typename Ta, typename Tc>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::transpose transa,
         oneapi::mkl::transpose transb, int m, int n, int k, int lda, int ldb, int ldc, Tc alpha,
         Tc beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    queue fpga_queue(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<Ta, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<Tc, usm::alloc::shared, 64>(cxt, *dev);
    vector<Ta, decltype(ua)> A(ua), B(ua);
    vector<Tc, decltype(uc)> C(ua);
    rand_matrix(A, layout, transa, m, k, lda);
    rand_matrix(B, layout, transb, k, n, ldb);
    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, m, n, ldc);

    auto C_ref = C;

    // Call MKL GEMM.
    oneapi::mkl::blas::row_major::gemm(main_queue, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
                                       ldb, beta, C_ref.data(), ldc, dependencies).wait();

    // Call T2SP GEMM
    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::gemm(fpga_queue, transa, transb, m, n, k,
                                                   alpha, A.data(), lda, B.data(), ldb, beta,
                                                   C.data(), ldc, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM:\n" << e.what() << std::endl;
        print_error_code(e);
    }
    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(C, C_ref, layout, m, n, ldc, 10 * k, std::cout);
    return (int)good;
}

class GemmUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(GemmUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#endif
}

TEST_P(GemmUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#endif
}

TEST_P(GemmUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_4)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_5)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_6)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_7)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_8)
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#endif
}

TEST_P(GemmUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_4)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_5)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_6)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_7)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#elif defined(T2SP_TEST_8)
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::conjtrans, 79, 84, 92, 103, 105, 106, alpha, beta)));
#endif
}

INSTANTIATE_TEST_SUITE_P(GemmUsmTestSuite, GemmUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
