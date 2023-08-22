#include <complex>
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

#include "api.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

sycl::device d{sycl::cpu_selector_v};
std::vector<sycl::device*> devices{&d};

namespace {

int test(device *dev, oneapi::mkl::layout layout, int N, int incx, int incy, float alpha) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during SDSDOT:\n"
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
    auto ua = usm_allocator<float, usm::alloc::shared, 64>(cxt, *dev);
    vector<float, decltype(ua)> x(ua), y(ua);
    float *result_reference = (float *)oneapi::mkl::malloc_shared(64, sizeof(float), *dev, cxt);

    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    // Call MKL SDSDOT.
    oneapi::mkl::blas::row_major::sdsdot(main_queue, N, alpha, x.data(), incx, y.data(),
                                         incy, result_reference, dependencies).wait();

    // Call T2SP SDSDOT.
    auto result_p = (float *)oneapi::mkl::malloc_shared(64, sizeof(float), *dev, cxt);
    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::sdsdot(fpga_queue, N, alpha, x.data(), incx, y.data(),
                                                     incy, result_p, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during SDSDOT:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of SDSDOT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal(*result_p, *result_reference, N, std::cout);
    oneapi::mkl::free_shared(result_p, cxt);
    oneapi::mkl::free_shared(result_reference, cxt);

    return (int)good;
}

class SdsdotUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(SdsdotUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2, 3, 2.0));
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -2, -3, 2.0));
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1, 1, 2.0));
}

INSTANTIATE_TEST_SUITE_P(SdsdotUsmTestSuite, SdsdotUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
