//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/opencl.h>
#include <stdio.h>
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
using namespace sycl;

constexpr int MAX_SOURCE_SIZE = 0x100000;
constexpr int N = 1024;

int main(int argc, char **argv) {
  size_t bytes = sizeof(float) * N;

  cl_float *host_a = (cl_float *)malloc(bytes);
  cl_float *host_b = (cl_float *)malloc(bytes);
  cl_float *host_c = (cl_float *)malloc(bytes);
  for (int i = 0; i < N; ++i) {
    host_a[i] = i;
    host_b[i] = i * 2;
  }

  FILE *fp;
  char *source_str;
  size_t source_size;
  fp = fopen("vector_add_kernel.cl", "r");
  if (!fp) {
    std::cerr << "Failed to load kernel file." << std::endl;
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  std::cout << "Kernel Loading Done" << std::endl;

  // Get platform and device information
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
  std::cout << "Platforms Found: " << ret_num_platforms << std::endl;

  cl_platform_id *ocl_platforms = (cl_platform_id *)malloc(ret_num_platforms);
  ret = clGetPlatformIDs(ret_num_platforms, ocl_platforms, NULL);
  // Set Platform to Use
  int platform_index = 0;
  platform sycl_platform = opencl::make<platform>(ocl_platforms[platform_index]);
  std::cout << "Using Platform: "
            << sycl_platform.get_info<info::platform::name>() << std::endl;

  ret = clGetDeviceIDs(ocl_platforms[platform_index], CL_DEVICE_TYPE_ALL, 1,
                       &device_id, &ret_num_devices);
  std::cout << "Devices Found: " << ret_num_devices << std::endl;

  // Create an OpenCL context and queue
  cl_context ocl_context =
      clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  cl_command_queue ocl_queue =
      clCreateCommandQueueWithProperties(ocl_context, device_id, 0, &ret);

  // Create a program from the kernel source, and build it
  cl_program ocl_program =
      clCreateProgramWithSource(ocl_context, 1, (const char **)&source_str,
                                (const size_t *)&source_size, &ret);
  ret = clBuildProgram(ocl_program, 1, &device_id, NULL, NULL, NULL);

  // OpenCL Kernel and Memory Objects
  cl_kernel ocl_kernel = clCreateKernel(ocl_program, "vector_add", &ret);
  cl_mem ocl_buf_a =
      clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  cl_mem ocl_buf_b =
      clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  cl_mem ocl_buf_c =
      clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  clEnqueueWriteBuffer(ocl_queue, ocl_buf_a, CL_TRUE, 0, bytes, host_a, 0, NULL,
                       NULL);
  clEnqueueWriteBuffer(ocl_queue, ocl_buf_b, CL_TRUE, 0, bytes, host_b, 0, NULL,
                       NULL);

  {  // DPC++ Application Scope
    // Construct SYCL versions of the context, queue, kernel, and buffers
    context sycl_context = opencl::make<context>(ocl_context);
    queue sycl_queue = opencl::make<queue>(sycl_context, ocl_queue);
    std::cout << "Device: "
              << sycl_queue.get_device().get_info<info::device::name>()
              << std::endl;
    kernel sycl_kernel = make_kernel<backend::opencl>(ocl_kernel, sycl_context);
    buffer<int, 1> sycl_buf_a = make_buffer<backend::opencl, int>(ocl_buf_a, sycl_context);
    buffer<int, 1> sycl_buf_b = make_buffer<backend::opencl, int>(ocl_buf_b, sycl_context);
    buffer<int, 1> sycl_buf_c = make_buffer<backend::opencl, int>(ocl_buf_c, sycl_context);
    sycl_queue.submit([&](handler &h) {
      // Create accessors for each of the buffers
      accessor a_accessor(sycl_buf_a, h, read_only);
      accessor b_accessor(sycl_buf_b, h, read_only);
      accessor c_accessor(sycl_buf_c, h, write_only);
      // Map kernel arguments to accessors
      h.set_args(a_accessor, b_accessor, c_accessor);
      // Launch Kernel
      h.parallel_for(range<1>(N), sycl_kernel);
    });
  }
  // Read buffer content back to host array
  clEnqueueReadBuffer(ocl_queue, ocl_buf_c, CL_TRUE, 0, bytes, host_c, 0, NULL,
                      NULL);

  for (int i = 0; i < N; ++i) {
    if (host_c[i] != i * 3) {
      std::cout << "Failed!" << std::endl;
      return -1;
    }
  }
  std::cout << "Passed!" << std::endl;

  ret = clReleaseCommandQueue(ocl_queue);
  ret = clReleaseKernel(ocl_kernel);
  ret = clReleaseProgram(ocl_program);
  ret = clReleaseMemObject(ocl_buf_a);
  ret = clReleaseMemObject(ocl_buf_b);
  ret = clReleaseMemObject(ocl_buf_c);
  ret = clReleaseContext(ocl_context);
  free(host_a);
  free(host_b);
  free(host_c);
  return 0;
}
