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

template <typename fp>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::side left_right,
         oneapi::mkl::uplo upper_lower, int m, int n, int lda, int ldb, int ldc, fp alpha,
         fp beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HEMM:\n"
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
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), B(ua), C(ua);
    rand_hermitian_matrix(A, layout, oneapi::mkl::transpose::nontrans, left_right == oneapi::mkl::side::left ? m : n, lda, true);
    rand_matrix(B, layout, oneapi::mkl::transpose::nontrans, m, n, ldb);
    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, m, n, ldc);

    auto C_ref = C;

    // Call MKL HEMM.
    oneapi::mkl::blas::row_major::hemm(main_queue, left_right, upper_lower, m, n,
                                       alpha, A.data(), lda, B.data(), ldb, beta,
                                       C_ref.data(), ldc, dependencies).wait();

    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::hemm(fpga_queue, left_right, upper_lower, m, n,
                                                   alpha, A.data(), lda, B.data(), ldb, beta,
                                                   C.data(), ldc, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HEMM:\n" << e.what() << std::endl;
        print_error_code(e);
    }
    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HEMM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(C, C_ref, layout, m, n, ldc, 10 * std::max(m, n), std::cout);
    return (int)good;
}

class HemmUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(HemmUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                                72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                                72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                                72, 28, 101, 102, 103, alpha, beta));
#endif
}
TEST_P(HemmUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                 72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                                 72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                                 72, 28, 101, 102, 103, alpha, beta));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                                 72, 28, 101, 102, 103, alpha, beta));
#endif
}

INSTANTIATE_TEST_SUITE_P(HemmUsmTestSuite, HemmUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
