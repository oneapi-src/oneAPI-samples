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

template <typename fp, typename fp_res>
int test(device *dev, oneapi::mkl::layout layout, int N, int incx) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during NRM2:\n"
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
    fp_res *result_reference = (fp_res *)oneapi::mkl::malloc_shared(64, sizeof(fp_res), *dev, cxt);

    rand_vector(x, N, incx);

    // Call MKL NRM2.
    oneapi::mkl::blas::row_major::nrm2(main_queue, N, x.data(), incx,
                                       result_reference, dependencies).wait();

    // Call T2SP NRM2.
    auto result_p = (fp_res *)oneapi::mkl::malloc_shared(64, sizeof(fp_res), *dev, cxt);
    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::nrm2(fpga_queue, N, x.data(), incx,
                                                   result_p, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during NRM2:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of NRM2:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_ptr(main_queue, result_p, *result_reference, N, std::cout);
    oneapi::mkl::free_shared(result_p, cxt);
    oneapi::mkl::free_shared(result_reference, cxt);

    return (int)good;
}

class Nrm2UsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(Nrm2UsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3)));
}
TEST_P(Nrm2UsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 2)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, 1)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1356, -3)));
}
TEST_P(Nrm2UsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1356, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1356, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1356, -3)));
}
TEST_P(Nrm2UsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1356, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1356, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1356, -3)));
}

INSTANTIATE_TEST_SUITE_P(Nrm2UsmTestSuite, Nrm2UsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
