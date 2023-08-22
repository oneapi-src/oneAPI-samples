#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "mkl_cblas.h"
#include <sycl/ext/intel/fpga_device_selector.hpp>
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

template <typename fp, typename fp_scalar>
int test(device *dev, oneapi::mkl::layout layout, int N, int incx, fp_scalar alpha) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during SCAL:\n"
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
    vector<fp, decltype(ua)> x(ua);

    rand_vector(x, N, incx);

    auto x_ref = x;

    // Call MKL SCAL.
    oneapi::mkl::blas::row_major::scal(main_queue, N, alpha, x_ref.data(), incx,
                                       dependencies).wait();

    // Call T2SP SCAL.
    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::scal(fpga_queue, N, alpha, x.data(), incx,
                                                   dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during SCAL:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of SCAL:\n" << error.what() << std::endl;
    }

    bool good = check_equal_vector(x, x_ref, N, incx, N, std::cout);

    return (int)good;
}

class ScalUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(ScalUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, alpha)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, alpha)));
}
TEST_P(ScalUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, alpha)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, alpha)));
}

INSTANTIATE_TEST_SUITE_P(ScalUsmTestSuite, ScalUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
