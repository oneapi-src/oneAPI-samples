//==============================================================
// Copyright © 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <unistd.h>
#include <sycl/sycl.hpp>

using namespace sycl;

static const char *usage_str =
    "\n ze_bandwidth [OPTIONS]"
    "\n"
    "\n OPTIONS:"
    "\n  -t, string               selectively run a particular test:"
    "\n      h2d or H2D                       run only Host-to-Device tests"
    "\n      d2h or D2H                       run only Device-to-Host tests "
    "\n                            [default:  both]"
    "\n  -q                       minimal output"
    "\n                            [default:  disabled]"
    "\n  -v                       enable verificaton"
    "\n                            [default:  disabled]"
    "\n  -i                       set number of iterations per transfer"
    "\n                            [default:  500]"
    "\n  -s                       select only one transfer size (bytes) "
    "\n  -sb                      select beginning transfer size (bytes)"
    "\n                            [default:  1]"
    "\n  -se                      select ending transfer size (bytes)"
    "\n                            [default: 2^28]"
    "\n  -l                       use SYCL prepare_for_device_copy/release_from_device_copy APIs"
    "\n                            [default: disabled]"
    "\n  -h, --help               display help message"
    "\n";

static uint32_t sanitize_ulong(char *in) {
  unsigned long temp = strtoul(in, NULL, 0);
  if (ERANGE == errno) {
    fprintf(stderr, "%s out of range of type ulong\n", in);
  } else if (temp > UINT32_MAX) {
    fprintf(stderr, "%ld greater than UINT32_MAX\n", temp);
  } else {
    return static_cast<uint32_t>(temp);
  }
  return 0;
}

size_t transfer_lower_limit = 1;
size_t transfer_upper_limit = (1 << 28);
bool verify = false;
bool run_host2dev = true;
bool run_dev2host = true;
bool verbose = true;
bool prepare = false;
uint32_t ntimes = 500;

//  kernel latency
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      std::cout << usage_str;
      exit(0);
    } else if (strcmp(argv[i], "-q") == 0) {
      verbose = false;
    } else if (strcmp(argv[i], "-v") == 0) {
      verify = true;
    } else if (strcmp(argv[i], "-l") == 0) {
      prepare = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      if ((i + 1) < argc) {
        ntimes = sanitize_ulong(argv[i + 1]);
        i++;
      }
    } else if (strcmp(argv[i], "-s") == 0) {
      if ((i + 1) < argc) {
        transfer_lower_limit = sanitize_ulong(argv[i + 1]);
        transfer_upper_limit = transfer_lower_limit;
        i++;
      }
    } else if (strcmp(argv[i], "-sb") == 0) {
      if ((i + 1) < argc) {
        transfer_lower_limit = sanitize_ulong(argv[i + 1]);
        i++;
      }
    } else if (strcmp(argv[i], "-se") == 0) {
      if ((i + 1) < argc) {
        transfer_upper_limit = sanitize_ulong(argv[i + 1]);
        i++;
      }
    } else if ((strcmp(argv[i], "-t") == 0)) {
      run_host2dev = false;
      run_dev2host = false;

      if ((i + 1) >= argc) {
        std::cout << usage_str;
        exit(-1);
      }
      if ((strcmp(argv[i + 1], "h2d") == 0) ||
          (strcmp(argv[i + 1], "H2D") == 0)) {
        run_host2dev = true;
        i++;
      } else if ((strcmp(argv[i + 1], "d2h") == 0) ||
                 (strcmp(argv[i + 1], "D2H") == 0)) {
        run_dev2host = true;
        i++;
      } else {
        std::cout << usage_str;
        exit(-1);
      }
    } else {
      std::cout << usage_str;
      exit(-1);
    }
  }

  queue dq;
  device dev = dq.get_device();
  size_t max_compute_units = dev.get_info<info::device::max_compute_units>();
  auto BE = dq.get_device()
                    .template get_info<sycl::info::device::opencl_c_version>()
                    .empty()
                ? "L0"
                : "OpenCL";
  if (verbose)
    std::cout << "Device name " << dev.get_info<info::device::name>() << " "
              << "max_compute units"
              << " " << max_compute_units << ", Backend " << BE << "\n";

  void *hostp;
  posix_memalign(&hostp, 4096, transfer_upper_limit);
  memset(hostp, 1, transfer_upper_limit);

  if (prepare) {
    if (verbose)
      std::cout << "Doing L0 Import\n";
    sycl::ext::oneapi::experimental::prepare_for_device_copy(
        hostp, transfer_upper_limit, dq);
  }

  void *destp =
      malloc_device<char>(transfer_upper_limit, dq.get_device(), dq.get_context());
  dq.submit([&](handler &cgh) { cgh.memset(destp, 2, transfer_upper_limit); });
  dq.wait();

  if (run_host2dev) {
    if (!verbose)
      printf("SYCL USM API (%s)\n", BE);
    for (size_t s = transfer_lower_limit; s <= transfer_upper_limit; s <<= 1) {
      auto start_time = std::chrono::steady_clock::now();
      for (int i = 0; i < ntimes; ++i) {
        dq.submit([&](handler &cgh) { cgh.memcpy(destp, hostp, s); });
        dq.wait();
      }
      auto end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> seconds = end_time - start_time;

      if (verbose)
        printf("HosttoDevice: %8lu bytes, %7.3f ms, %8.3g GB/s\n", s,
               1000 * seconds.count() / ntimes,
               1e-9 * s / (seconds.count() / ntimes));
      else
        printf("%10.6f\n", 1e-9 * s / (seconds.count() / ntimes));
    }
  }

  if (run_dev2host) {
    if (!verbose)
      printf("SYCL USM API (%s)\n", BE);
    for (size_t s = transfer_lower_limit; s <= transfer_upper_limit; s <<= 1) {
      auto start_time = std::chrono::steady_clock::now();
      for (int i = 0; i < ntimes; ++i) {
        dq.submit([&](handler &cgh) { cgh.memcpy(hostp, destp, s); });
        dq.wait();
      }
      auto end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> seconds = end_time - start_time;

      if (verbose)
        printf("DeviceToHost: %8lu bytes, %7.3f ms, %8.3g GB/s\n", s,
               seconds.count(), 1e-9 * s / (seconds.count() / ntimes));
      else
        printf("%10.6f\n", 1e-9 * s / (seconds.count() / ntimes));
    }
  }

  if (prepare)
    sycl::ext::oneapi::experimental::release_from_device_copy(hostp, dq);

  free(hostp);
  free(destp, dq.get_context());
}
// Snippet end
