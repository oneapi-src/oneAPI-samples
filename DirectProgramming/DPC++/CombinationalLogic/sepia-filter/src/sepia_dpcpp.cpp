//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <chrono>
#include <cmath>
#include <iostream>
#include "CL/sycl.hpp"
#include "device_selector.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

// stb/*.h files can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/stb/*.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;
using namespace sycl;

// Few useful acronyms.
constexpr auto sycl_read = access::mode::read;
constexpr auto sycl_write = access::mode::write;
constexpr auto sycl_global_buffer = access::target::global_buffer;

static void ReportTime(const string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();

  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  cout << msg << elapsed << " milliseconds\n";
}

// SYCL does not need any special mark-up for functions which are called from
// SYCL kernel and defined in the same compilation unit. SYCL compiler must be
// able to find the full call graph automatically.
// always_inline as calls are expensive on Gen GPU.
// Notes:
// - coeffs can be declared outside of the function, but still must be constant
// - SYCL compiler will automatically deduce the address space for the two
//   pointers; sycl::multi_ptr specialization for particular address space
//   can used for more control
__attribute__((always_inline)) static void ApplyFilter(uint8_t *src_image,
                                                       uint8_t *dst_image,
                                                       int i) {
  i *= 3;
  float temp;
  temp = (0.393f * src_image[i]) + (0.769f * src_image[i + 1]) +
         (0.189f * src_image[i + 2]);
  dst_image[i] = temp > 255 ? 255 : temp;
  temp = (0.349f * src_image[i]) + (0.686f * src_image[i + 1]) +
         (0.168f * src_image[i + 2]);
  dst_image[i + 1] = temp > 255 ? 255 : temp;
  temp = (0.272f * src_image[i]) + (0.534f * src_image[i + 1]) +
         (0.131f * src_image[i + 2]);
  dst_image[i + 2] = temp > 255 ? 255 : temp;
}

// This is alternative (to a lambda) representation of a SYCL kernel.
// Internally, compiler transforms lambdas into instances of a very simlar
// class. With functors, capturing kernel parameters is done manually via the
// constructor, unlike automatic capturing with lambdas.
class SepiaFunctor {
 public:
  // Constructor captures needed data into fields
  SepiaFunctor(
      accessor<uint8_t, 1, sycl_read, sycl_global_buffer> &image_acc_,
      accessor<uint8_t, 1, sycl_write, sycl_global_buffer> &image_exp_acc_)
      : image_acc(image_acc_), image_exp_acc(image_exp_acc_) {}

  // The '()' operator is the actual kernel
  void operator()(id<1> i) const {
    ApplyFilter(image_acc.get_pointer(), image_exp_acc.get_pointer(), i.get(0));
  }

 private:
  // Captured values:
  accessor<uint8_t, 1, sycl_read, sycl_global_buffer> image_acc;
  accessor<uint8_t, 1, sycl_write, sycl_global_buffer> image_exp_acc;
};

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Program usage is <executable> <inputfile>\n";
    exit(1);
  }

  // loading the input image
  int img_width, img_height, channels;
  uint8_t *image = stbi_load(argv[1], &img_width, &img_height, &channels, 0);
  if (image == NULL) {
    cout << "Error in loading the image\n";
    exit(1);
  }
  cout << "Loaded image with a width of " << img_width << ", a height of "
            << img_height << " and " << channels << " channels\n";

  size_t num_pixels = img_width * img_height;
  size_t img_size = img_width * img_height * channels;

  // allocating memory for output images
  uint8_t *image_ref = new uint8_t[img_size];
  uint8_t *image_exp1 = new uint8_t[img_size];
  uint8_t *image_exp2 = new uint8_t[img_size];

  memset(image_ref, 0, img_size * sizeof(uint8_t));
  memset(image_exp1, 0, img_size * sizeof(uint8_t));
  memset(image_exp2, 0, img_size * sizeof(uint8_t));

  // Create a device selector which rates available devices in the preferred
  // order for the runtime to select the highest rated device
  // Note: This is only to illustrate the usage of a custom device selector.
  // default_selector can be used if no customization is required.
  MyDeviceSelector sel;

  // Using these events to time command group execution
  event e1, e2;

  // Wrap main SYCL API calls into a try/catch to diagnose potential errors
  try {
    // Create a command queue using the device selector and request profiling
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(sel, dpc_common::exception_handler, prop_list);

    // See what device was actually selected for this queue.
    cout << "Running on " << q.get_device().get_info<info::device::name>()
              << "\n";

    // Create SYCL buffer representing source data .
    // By default, this buffers will be created with global_buffer access
    // target, which means the buffer "projection" to the device (actual
    // device memory chunk allocated or mapped on the device to reflect
    // buffer's data) will belong to the SYCL global address space - this
    // is what host data usually maps to. Other address spaces are:
    // private, local and constant.
    // Notes:
    // - access type (read/write) is not specified when creating a buffer -
    //   this is done when actual accessor is created
    // - there can be multiple accessors to the same buffer in multiple command
    //   groups
    // - 'image' pointer was passed to the constructor, so this host memory
    //   will be used for "host projection", no allocation will happen on host
    buffer image_buf(image, range(img_size));

    // This is the output buffer device writes to
    buffer image_buf_exp1(image_exp1, range(img_size));
    cout << "Submitting lambda kernel...\n";

    // Submit a command group for execution. Returns immediately, not waiting
    // for command group completion.
    e1 = q.submit([&](handler &h) {
      // This lambda defines a "command group" - a set of commands for the
      // device sharing some state and executed in-order - i.e. creation of
      // accessors may lead to on-device memory allocation, only after that
      // the kernel will be enqueued.
      // A command group can contain at most one parallel_for, single_task or
      // parallel_for_workgroup construct.
      auto image_acc = image_buf.get_access<sycl_read>(h);
      auto image_exp_acc = image_buf_exp1.get_access<sycl_write>(h);

      // This is the simplest form cl::sycl::handler::parallel_for -
      // - it specifies "flat" 1D ND range(num_pixels), runtime will select
      //   local size
      // - kernel lambda accepts single cl::sycl::id argument, which has very
      //   limited API; see the spec for more complex forms
      // the lambda parameter of the parallel_for is the kernel, which
      // actually executes on device
      h.parallel_for(range<1>(num_pixels), [=](id<1> i) {
        ApplyFilter(image_acc.get_pointer(), image_exp_acc.get_pointer(),
                    i.get(0));
      });
    });
    q.wait_and_throw();

    cout << "Submitting functor kernel...\n";

    buffer image_buf_exp2(image_exp2, range(img_size));

    // Submit another command group. This time kernel is represented as a
    // functor object.
    e2 = q.submit([&](handler &h) {
      auto image_acc = image_buf.get_access<sycl_read>(h);
      auto image_exp_acc = image_buf_exp2.get_access<sycl_write>(h);

      SepiaFunctor kernel(image_acc, image_exp_acc);

      h.parallel_for(range<1>(num_pixels), kernel);
    });

    cout << "Waiting for execution to complete...\n";

    q.wait_and_throw();

  } catch (sycl::exception e) {
    // This catches only synchronous exceptions that happened in current thread
    // during execution. The asynchronous exceptions caused by execution of the
    // command group are caught by the asynchronous exception handler
    // registered. Synchronous exceptions are usually those which are thrown
    // from the SYCL runtime code, such as on invalid constructor arguments. An
    // example of asynchronous exceptions is error occurred during execution of
    // a kernel. Make sure sycl::exception is caught, not std::exception.
    cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }

  cout << "Execution completed\n";

  // report execution times:
  ReportTime("Lambda kernel time: ", e1);
  ReportTime("Functor kernel time: ", e2);

  // get reference result
  for (size_t i = 0; i < num_pixels; i++) {
    ApplyFilter(image, image_ref, i);
  }

  stbi_write_png("sepia_ref.png", img_width, img_height, channels, image_ref,
                 img_width * channels);
  stbi_write_png("sepia_lambda.png", img_width, img_height, channels,
                 image_exp1, img_width * channels);
  stbi_write_png("sepia_functor.png", img_width, img_height, channels,
                 image_exp2, img_width * channels);

  stbi_image_free(image);
  delete[] image_ref;
  delete[] image_exp1;
  delete[] image_exp2;
  
  cout << "Sepia tone successfully applied to image:[" << argv[1] << "]\n";
  return 0;
}
