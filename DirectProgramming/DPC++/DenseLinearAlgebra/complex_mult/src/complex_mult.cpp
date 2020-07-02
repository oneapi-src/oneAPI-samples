
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <vector>
#include "Complex.hpp"
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

// Number of complex numbers passing to the DPC++ code
static const int num_elements = 16;

class custom_device_selector : public device_selector {
 public:
  custom_device_selector(std::string vendorName) : vendorName_(vendorName){};
  int operator()(const device &dev) const override {
    int rating = 0;
    // In the below code we are querying for the custom device specific to a
    // Vendor and if it is a GPU device we are giving the highest rating. The
    // second preference is given to any GPU device and the third preference is
    // given to CPU device.
    if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) !=
                        std::string::npos))
      rating = 3;
    else if (dev.is_gpu())
      rating = 2;
    else if (dev.is_cpu())
      rating = 1;
    return rating;
  };

 private:
  std::string vendorName_;
};

// in_vect1 and in_vect2 are the vectors with num_elements complex nubers and
// are inputs to the parallel function
void DpcppParallel(queue &q, std::vector<Complex2> &in_vect1,
                   std::vector<Complex2> &in_vect2,
                   std::vector<Complex2> &out_vect) {
  auto R = range(num_elements);
  // Setup input buffers
  buffer bufin_vect1(in_vect1);
  buffer bufin_vect2(in_vect2);

  // Setup Output buffers
  buffer bufout_vect(out_vect.data(), R);

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  // Submit Command group function object to the queue
  q.submit([&](auto &h) {
    // Accessors set as read mode
    auto V1 = bufin_vect1.get_access<access::mode::read>(h);
    auto V2 = bufin_vect2.get_access<access::mode::read>(h);
    // Accessor set to Write mode
    auto V3 = bufout_vect.get_access<access::mode::write>(h);

    h.parallel_for(R, [=](id<1> i) {
      // call the complex_mul function that computes the multiplication of the
      // complex numberr
      V3[i] = V1[i].complex_mul(V2[i]);
    });
  });
  q.wait_and_throw();
}
void DpcppScalar(std::vector<Complex2> &in_vect1,
                 std::vector<Complex2> &in_vect2,
                 std::vector<Complex2> &out_vect) {
  for (int i = 0; i < num_elements; i++) {
    out_vect[i] = in_vect1[i].complex_mul(in_vect2[i]);
  }
}
// Compare the results of the two output vectors from parallel and scalar. They
// should be equal
int Compare(std::vector<Complex2> &v1, std::vector<Complex2> &v2) {
  int ret_code = 1;
  for (int i = 0; i < num_elements; i++) {
    if (v1[i] != v2[i]) {
      ret_code = -1;
      break;
    }
  }
  return ret_code;
}
int main() {
  // Declare your Input and Output vectors of the Complex2 class
  vector<Complex2> input_vect1;
  vector<Complex2> input_vect2;
  vector<Complex2> out_vect_parallel;
  vector<Complex2> out_vect_scalar;

  for (int i = 0; i < num_elements; i++) {
    input_vect1.push_back(Complex2(i + 2, i + 4));
    input_vect2.push_back(Complex2(i + 4, i + 6));
    out_vect_parallel.push_back(Complex2(0, 0));
    out_vect_scalar.push_back(Complex2(0, 0));
  }

  // Initialize your Input and Output Vectors. Inputs are initialized as below.
  // Outputs are initialized with 0
  try {
    // Pass in the name of the vendor for which the device you want to query
    std::string vendor_name = "Intel";
    // std::string vendor_name = "AMD";
    // std::string vendor_name = "Nvidia";
    // queue constructor passed exception handler
    custom_device_selector selector(vendor_name);
    queue q(selector, dpc::exception_handler);
    // Call the DpcppParallel with the required inputs and outputs
    DpcppParallel(q, input_vect1, input_vect2, out_vect_parallel);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  cout << "****************************************Multiplying Complex numbers "
          "in Parallel********************************************************"
       << std::endl;
  // Print the outputs of the Parallel function
  for (int i = 0; i < num_elements; i++) {
    cout << out_vect_parallel[i] << ' ';
    if (i == num_elements - 1) {
      cout << "\n\n";
    }
  }
  cout << "****************************************Multiplying Complex numbers "
          "in Serial***********************************************************"
       << std::endl;
  // Call the DpcppScalar function with the required input and outputs
  DpcppScalar(input_vect1, input_vect2, out_vect_scalar);
  for (auto it = out_vect_scalar.begin(); it != out_vect_scalar.end(); it++) {
    cout << *it << ' ';
    if (it == out_vect_scalar.end() - 1) {
      cout << "\n\n";
    }
  }

  // Compare the outputs from the parallel and the scalar functions. They should
  // be equal

  int ret_code = Compare(out_vect_parallel, out_vect_scalar);
  if (ret_code == 1) {
    cout << "********************************************Success. Results are "
            "matched******************************"
         << "\n";
  } else
    cout << "*********************************************Failed. Results are "
            "not matched**************************"
         << "\n";

  return 0;
}
