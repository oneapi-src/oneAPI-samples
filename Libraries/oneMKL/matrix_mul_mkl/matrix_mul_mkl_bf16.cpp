// Matrix Multiplication utilizing the AMX/XMX hardware with
//   oneAPI Math Kernel Library.
// We use the bfloat16 type to fill in the matrices and on 
//   invoking the gemm function call appropriate mkl internal
//   functions having AMX/XMX instruction support are called.

#include <exception>
#include <iostream>

#include <CL/sycl.hpp>
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


    // Create a queue on the default device.
    sycl::queue my_queue{sycl::default_selector()};

	// create sycl buffers of matrix data for offloading between device and host
    sycl::buffer<bfloat16, 1> A_buffer(A.data(), A.size());
    sycl::buffer<bfloat16, 1> B_buffer(B.data(), B.size());
    sycl::buffer<bfloat16, 1> C_buffer(C.data(), C.size());

    try {
        using oneapi::mkl::blas::gemm;
        using oneapi::mkl::transpose;
        gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A_buffer, ldA, B_buffer,
           ldB, beta, C_buffer, ldC);
    
		my_queue.wait_and_throw();
    }
    catch (sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
            << e.what() << std::endl;
    }
    catch (std::exception const& e) {
        std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
            << e.what() << std::endl;
    }

	// post process results
    //
    // Access data from C buffer and print out part of C matrix
    auto C_accessor = C_buffer.template get_access<sycl::access::mode::read>();
    std::cout << "C" << " = [ " << C_accessor[0] << ", "
        << C_accessor[1] << ", ... ]\n";
    std::cout << "    [ " << C_accessor[1 * ldC + 0] << ", "
        << C_accessor[1 * ldC + 1] << ",  ... ]\n";
    std::cout << "    [ " << "... ]\n";
    std::cout << std::endl;


    return 0;
}
