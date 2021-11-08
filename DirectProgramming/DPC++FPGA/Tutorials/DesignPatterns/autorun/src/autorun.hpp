#ifndef __AUTORUN_HPP__
#define __AUTORUN_HPP__

#include <CL/sycl.hpp>
#include <type_traits>

/*
This header defines the Autorun kernel utility. This utility is used to
launch kernels that are submitted before main begins. It is typically used
to launch kernels that run forever.

Two classes are defined in this header file: Autorun and AutorunForever.
Autorun creates an autorun kernel that is NOT implicitly wrapped in an infinite
loop.
AutorunForever creates an autorun kernel that is implicitly wrapped in an
infinite loop.

The following describes the common template and constructor arguments for both
the Autorun and AutorunForever.

Template Args:
  KernelID (optional): the name of the autorun kernel.
  DeviceSelector: The type of the device selector.
  KernelFunctor: the kernel functor type.
Constructor Arguments:
    device_selector: the SYCL device selector
    kernel: the user-defined kernel functor.
            This defines the logic of the autorun kernel.
*/

namespace fpga_tools {
  //
  // Autorun kernel wrapper.
  // If the user wants the kernel to run forever, they must explicitly add a
  // while(1) loop into their kernel code.
  //
  template<typename KernelID = void>
  struct Autorun {
    // Constructor with a kernel name
    template<typename DeviceSelector, typename KernelFunctor,
             typename KernelID2 = KernelID>
    Autorun(DeviceSelector device_selector, KernelFunctor kernel,
            typename std::enable_if_t<!std::is_same_v<KernelID2, void>> *
            = nullptr) {
      // static asserts to ensure KernelFunctor is callable
      static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                   "KernelFunctor must be callable with no arguments");

      // create the device queue
      sycl::queue q{device_selector};

      // run the single task kernel
      q.single_task<KernelID>(kernel);
    }

    // Constructor without a kernel name
    template<typename DeviceSelector, typename KernelFunctor,
             typename KernelID2 = KernelID>
    Autorun(DeviceSelector device_selector, KernelFunctor kernel,
            typename std::enable_if_t<std::is_same_v<KernelID2, void>> *
            = nullptr) {
      // static asserts to ensure KernelFunctor is callable
      static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                   "KernelFunctor must be callable with no arguments");

      // create the device queue
      sycl::queue q{device_selector};

      // run the single task kernel
      q.single_task(kernel);
    }
  };

  //
  // AutorunForever kernel wrapper.
  // The user's code is implicitly wrapped in an infinite wile(1) loop.
  //
  template<typename KernelID = void>
  struct AutorunForever {
    // Constructor with a kernel name
    template<typename DeviceSelector, typename KernelFunctor,
             typename KernelID2 = KernelID>
    AutorunForever(DeviceSelector device_selector, KernelFunctor kernel,
                   typename std::enable_if_t<!std::is_same_v<KernelID2, void>> *
                   = nullptr) {
      // static asserts to ensure KernelFunctor is callable
      static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                   "KernelFunctor must be callable with no arguments");
      
      // create the device queue
      sycl::queue q{device_selector};

      // run the single task
      q.single_task<KernelID>([=] {
        while (1) {
          kernel();
        }
      });
    }

    // Constructor without a kernel name
    template<typename DeviceSelector, typename KernelFunctor,
             typename KernelID2 = KernelID>
    AutorunForever(DeviceSelector device_selector, KernelFunctor kernel,
                   typename std::enable_if_t<std::is_same_v<KernelID2, void>> *
                   = nullptr) {
      // static asserts to ensure KernelFunctor is callable
      static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                   "KernelFunctor must be callable with no arguments");
      
      // create the device queue
      sycl::queue q{device_selector};

      // run the single task
      q.single_task([=] {
        while (1) {
          kernel();
        }
      });
    }

  };
}  // namespace fpga_tools


#endif /* __AUTORUN_HPP__ */