#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The sdsdot API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"

using namespace std;

void test(int N, int incx, int incy) {
    vector<float, allocator_helper<float, 64>> x, y;
    float res{};
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    auto res_ref = res;

    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler, sycl::property::queue::enable_profiling());

    auto done = t2sp::blas::row_major::sdsdot(q_device, N, rand_scalar<float>(), x.data(), incx, y.data(),
                                              incy, &res);
    done.wait();

    // Get time in ns
    uint64_t start = done.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = done.template get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    double number_ops = 2.0 * N;
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of vector x: " << N << "\n";
    std::cout << "Size of vector y: " << N << "\n";
}

int main() {
    using test_type = double;
    const auto KKK = t2sp::blas::row_major::get_systolic_array_dimensions<test_type>();
    int64_t n = KKK * 4096 * 4096;
    test(n, 1, 1);
}
