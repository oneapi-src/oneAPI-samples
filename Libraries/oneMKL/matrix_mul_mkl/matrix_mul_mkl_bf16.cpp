// Standard SYCL header
#include <CL/sycl.hpp>
// STL classes
#include <exception>
#include <iostream>
// Declarations for Intel oneAPI Math Kernel Library SYCL/DPC++ APIs
#include "oneapi/mkl.hpp"
int main(int argc, char *argv[]) {

	// User obtains data here for A, B, C matrices, along with setting m, n, k, ldA, ldB, ldC.
    // For this example, A, B and C should be initially stored in a std::vector,
    //   or a similar container having data() and size() member functions.

using namespace oneapi::mkl;
int64_t m = 10, n = 6, k = 8, ldA = 12, ldB = 8, ldC = 10;
int64_t sizea = ldA * k, sizeb = ldB * n, sizec = ldC * n;
double alpha = 1.0, beta = 0.0;
// Allocate matrices
std::vector<bfloat16> A(sizea);
std::vector<bfloat16> B(sizeb);
std::vector<bfloat16> C(sizec);
// Initialize matrices […]
std::fill(A.begin(), A.end(), 0);
std::fill(B.begin(), B.end(), 0);
std::fill(C.begin(), C.end(), 0);


    sycl::device my_device = sycl::device(sycl::default_selector());
    // Create asynchronous exceptions handler to be attached to queue.
    // Not required; can provide helpful information in case the system isn’t correctly configured.
    auto my_exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n"
                    << e.what() << std::endl;
            }
            catch (std::exception const& e) {
                std::cout << "Caught asynchronous STL exception:\n"
                    << e.what() << std::endl;
            }
        }
    };
    // create execution queue with exception handler attached
    sycl::queue my_queue(my_device, my_exception_handler);
    // create sycl buffers of matrix data for offloading between device and host
    sycl::buffer<bfloat16, 1> A_buffer(A.data(), A.size());
    sycl::buffer<bfloat16, 1> B_buffer(B.data(), B.size());
    sycl::buffer<bfloat16, 1> C_buffer(C.data(), C.size());
    // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions
    try {
        using oneapi::mkl::blas::gemm;
        using oneapi::mkl::transpose;
        gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A_buffer, ldA, B_buffer,
           ldB, beta, C_buffer, ldC);
    }
    catch (sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
            << e.what() << std::endl;
    }
    catch (std::exception const& e) {
        std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
            << e.what() << std::endl;
    }
    // ensure any asynchronous exceptions caught are handled before proceeding
    my_queue.wait_and_throw();
    //
    // post process results
    //
    // Access data from C buffer and print out part of C matrix
    auto C_accessor = C_buffer.template get_access<sycl::access::mode::read>();
    std::cout << "\t" << "C" << " = [ " << C_accessor[0] << ", "
        << C_accessor[1] << ", ... ]\n";
    std::cout << "\t    [ " << C_accessor[1 * ldC + 0] << ", "
        << C_accessor[1 * ldC + 1] << ",  ... ]\n";
    std::cout << "\t    [ " << "... ]\n";
    std::cout << std::endl;


    return 0;
}
