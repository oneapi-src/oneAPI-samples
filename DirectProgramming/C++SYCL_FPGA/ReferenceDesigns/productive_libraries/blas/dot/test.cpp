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

template <typename fp, typename fp_res, usm::alloc alloc_type = usm::alloc::shared>
int test(device* dev, oneapi::mkl::layout layout, int N, int incx, int incy) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during DOT:\n"
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
    vector<fp, decltype(ua)> x(ua), y(ua);
    fp_res *result_ref = (fp_res*)oneapi::mkl::malloc_shared(64, sizeof(fp_res), *dev, cxt);

    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    // Call MKL DOT.
    oneapi::mkl::blas::row_major::dot(main_queue, N, x.data(), incx, y.data(),
                                      incy, result_ref, dependencies).wait();
    // Call T2SP DOT.
    fp_res* result_p;
    if constexpr (alloc_type == usm::alloc::shared) {
        result_p = (fp_res*)oneapi::mkl::malloc_shared(64, sizeof(fp_res), *dev, cxt);
    }
    else if constexpr (alloc_type == usm::alloc::device) {
        result_p = (fp_res*)oneapi::mkl::malloc_device(64, sizeof(fp_res), *dev, cxt);
    }
    else {
        throw std::runtime_error("Bad alloc_type");
    }

    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::dot(fpga_queue, N, x.data(), incx, y.data(),
                                                  incy, result_p, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during DOT:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of DOT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_ptr(main_queue, result_p, *result_ref, N, std::cout);

    oneapi::mkl::free_usm(result_p, cxt);
    oneapi::mkl::free_usm(result_ref, cxt);

    return (int)good;
}

class DotUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(DotUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, 3)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 100, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, -2)));
}

TEST_P(DotUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, 3)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 100, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, -2)));
}

TEST_P(DotUsmTests, RealDoubleSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, 3)));
    EXPECT_TRUEORSKIP(
        (test<float, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 100, 1, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3, -2)));
}

INSTANTIATE_TEST_SUITE_P(DotUsmTestSuite, DotUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
