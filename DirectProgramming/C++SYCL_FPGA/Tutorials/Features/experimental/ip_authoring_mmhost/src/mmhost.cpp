#include <iomanip>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using namespace sycl;
using ext::intel::experimental::property::usm::buffer_location;

template <typename T>
inline std::shared_ptr<T> make_malloc_shared(queue &q, int n, int kBL) {
  T *mem =
      malloc_shared<T>(sizeof(T) * n, q, property_list{buffer_location(kBL)});
  return std::move(
      std::shared_ptr<T>(mem, [&q](T *ptr) { sycl::free(ptr, q); }));
}

constexpr int BL1 = 1;
constexpr int BL2 = 2;
constexpr int BL3 = 3;

// Using pointers,
struct PointerIP {
  int *x, *y, *z;
  int size;

  PointerIP(int *x_, int *y_, int *z_, int size_)
      : x(x_), y(y_), z(z_), size(size_) {}

  void operator()() const {
    for (int i = 0; i < size; ++i) {
      int mul = x[i] * y[i];
      int add = mul + z[i];
      y[i] = mul;
      z[i] = add;
    }
  }
};

// Using mmhost macro
struct VectorMADIP {
  mmhost(BL1,  // buffer_location or aspace
         28,   // address width
         64,   // data width
         16,   // latency
         1,    // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,    // maxburst
         0,    // align, 0 defaults to alignment of the type
         1     // waitrequest, 0: false, 1: true
         ) int *x;
  mmhost(BL2,  // buffer_location or aspace
         28,   // address width
         64,   // data width
         16,   // ! latency, must be atleast 16
         0,    // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,    // maxburst
         0,    // align, 0 defaults to alignment of the type
         1     // waitrequest, 0: false, 1: true
         ) int *y;
  mmhost(BL3,  // buffer_location or aspace
         28,   // address width
         64,   // data width
         16,   // ! latency, must be atleast 16
         0,    // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,    // maxburst
         0,    // align, 0 defaults to alignment of the type
         1     // waitrequest, 0: false, 1: true
         ) int *z;
  int size;

  VectorMADIP(int *x_, int *y_, int *z_, int size_)
      : x(x_), y(y_), z(z_), size(size_) {}

  void operator()() const {
    for (int i = 0; i < size; ++i) {
      int mul = x[i] * y[i];
      int add = mul + z[i];
      y[i] = mul;
      z[i] = add;
    }
  }
};

void fillArrays(int *x, int *y, int *z, int size) {
  for (int i = 0; i < size; ++i) {
    x[i] = i;
    y[i] = i * 2;
    z[i] = i * 3;
  }
}

int main(void) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    sycl::queue q(selector, fpga_tools::exception_handler,
                  property::queue::enable_profiling{});

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    int size = 10000;
    double start, end;
    event e;

    // Allocate memory for pointer kernel arguments with specified buffer location
    auto x = make_malloc_shared<int>(q, size, BL1);
    auto y = make_malloc_shared<int>(q, size, BL2);
    auto z = make_malloc_shared<int>(q, size, BL3);

    fillArrays(x.get(), y.get(), z.get(), size);

    e = q.single_task(VectorMADIP{x.get(), y.get(), z.get(), size});
    e.wait();

    start = e.get_profiling_info<info::event_profiling::command_start>();
    end = e.get_profiling_info<info::event_profiling::command_end>();
    // convert from nanoseconds to ms
    double kernel_mmhost_time = (double)(end - start) * 1e-6;

```suggestion
    // Allocate memory for pointer kernel arguments, no buffer location is specified
    auto px = malloc_shared<int>(size, q);
    auto py = malloc_shared<int>(size, q);
    auto pz = malloc_shared<int>(size, q);

    fillArrays(px, py, pz, size);
    e = q.single_task(PointerIP{px, py, pz, size});
    e.wait();

    start = e.get_profiling_info<info::event_profiling::command_start>();
    end = e.get_profiling_info<info::event_profiling::command_end>();
    // convert from nanoseconds to ms
    double kernel_pointer_time = (double)(end - start) * 1e-6;

    std::cout << "MMHost kernel time : " << kernel_mmhost_time << " ms\n";
    std::cout << "Pointer kernel time : " << kernel_pointer_time << " ms\n";
    std::cout << "elements in vector : " << size << "\n";

    bool pass_check = true;
    for (int i = 0; i < size; ++i) {
      int mul = i * i * 2;
      int add = mul + i * 3;
      if (x.get()[i] != i || y.get()[i] != mul || z.get()[i] != add) {
        pass_check = false;
        break;
      }
    }
    for (int i = 0; i < size; ++i) {
      int mul = i * i * 2;
      int add = mul + i * 3;
      if (px[i] != i || py[i] != mul || pz[i] != add) {
        pass_check = false;
        break;
      }
    }

    if (!pass_check) {
      std::cout << "--> FAIL\n";
    } else {
      std::cout << "--> PASS\n";
    }
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}