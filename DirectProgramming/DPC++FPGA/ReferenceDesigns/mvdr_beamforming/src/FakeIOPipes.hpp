#ifndef __FAKEIOPIPES_HPP__
#define __FAKEIOPIPES_HPP__

#include <iostream>
#include <type_traits>
#include <utility>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// the "detail" namespace is commonly used in C++ as an internal namespace
// (to a file) that is not meant to be visible to the public and should be
// ignored by external users. That is to say, you should never have the line:
// "using namespace detail;" in your code!
//
// "internal" is another common name for a namespace like this.
namespace detail {

using namespace sycl;

template <typename Id, typename T, bool use_host_alloc>
class ProducerConsumerBaseImpl {
 protected:
  // private members
  static inline T *host_data_{nullptr};
  static inline T *device_data_{nullptr};
  static inline size_t count_{};
  static inline bool initialized_{false};

  // use some fancy C++ metaprogramming to get the correct pointer type
  // based on the template variable
  typedef
      typename std::conditional_t<use_host_alloc, host_ptr<T>, device_ptr<T>>
          kernel_ptr_type;

  // private constructor so users cannot make an object
  ProducerConsumerBaseImpl(){};

  static T *get_kernel_ptr() {
    return use_host_alloc ? host_data_ : device_data_;
  }

  static void initialized_check() {
    if (!initialized_) {
      std::cerr << "ERROR: Init() has not been called\n";
      std::terminate();
    }
  }

 public:
  // disable copy constructor and operator=
  ProducerConsumerBaseImpl(const ProducerConsumerBaseImpl &) = delete;
  ProducerConsumerBaseImpl &operator=(ProducerConsumerBaseImpl const &) =
      delete;

  static void Init(queue &q, size_t count) {
    // make sure init hasn't already been called
    if (initialized_) {
      std::cerr << "ERROR: Init() was already called\n";
      std::terminate();
    }

    // track count
    count_ = count;

    // check for USM support
    device d = q.get_device();
    if (!d.get_info<info::device::usm_host_allocations>() && use_host_alloc) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      std::terminate();
    }
    if (!d.get_info<info::device::usm_device_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM device"
                << " allocations\n";
      std::terminate();
    }

    // Allocate the space the user requested. Calling a different malloc
    // based on whether the user wants to use USM host allocations or not.
    if (use_host_alloc) {
      host_data_ = malloc_host<T>(count_, q);
    } else {
      host_data_ = new T[count_];
    }

    if (host_data_ == nullptr) {
      std::cerr << "ERROR: failed to allocate space for host_data_\n";
      std::terminate();
    }

    // if not using host allocations, allocate device memory
    if (!use_host_alloc) {
      device_data_ = malloc_device<T>(count_, q);
      if (device_data_ == nullptr) {
        std::cerr << "ERROR: failed to allocate space for"
                  << "device_data_\n";
        std::terminate();
      }
    }

    initialized_ = true;
  }

  static void Destroy(queue &q) {
    initialized_check();

    // free memory depending on 'use_host_alloc' flag
    if (use_host_alloc) {
      // free USM host allocation
      sycl::free(host_data_, q);
    } else {
      // free C++ allocated memory
      delete[] host_data_;

      // free USM device allocation
      sycl::free(device_data_, q);
    }

    initialized_ = false;
  }

  static size_t Count() {
    initialized_check();
    return count_;
  }

  static T *Data() {
    initialized_check();
    return host_data_;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Producer implementation
template <typename Id, typename T, bool use_host_alloc, size_t min_capacity>
class ProducerImpl : public ProducerConsumerBaseImpl<Id, T, use_host_alloc> {
 private:
  // base implementation alias
  using BaseImpl = ProducerConsumerBaseImpl<Id, T, use_host_alloc>;
  using kernel_ptr_type = typename BaseImpl::kernel_ptr_type;

  // IDs for the pipe and kernel
  class PipeID;
  class KernelID;

  // private constructor so users cannot make an object
  ProducerImpl(){};

 public:
  // disable copy constructor and operator=
  ProducerImpl(const ProducerImpl &) = delete;
  ProducerImpl &operator=(ProducerImpl const &) = delete;

  // the pipe to connect to in device code
  using Pipe = sycl::INTEL::pipe<PipeID, T, min_capacity>;

  // the implementation of the static
  static std::pair<event, event> Start(queue &q,
                                       size_t count = BaseImpl::count_) {
    // make sure initialized has been called
    BaseImpl::initialized_check();

    // can't produce more data than exists
    if (count > BaseImpl::count_) {
      std::cerr << "ERROR: Start() called with count=" << count
                << " but allocated size is " << BaseImpl::count_ << "\n";
      std::terminate();
    }

    // If we aren't using USM host allocations, must transfer memory to device
    event dma_event;
    if (!use_host_alloc) {
      dma_event = q.memcpy(BaseImpl::device_data_, BaseImpl::host_data_,
                           BaseImpl::count_ * sizeof(T));
    }

    // pick the right pointer to pass to the kernel
    auto kernel_ptr = BaseImpl::get_kernel_ptr();

    // launch the kernel (use event.depends_on to wait on the memcpy)
    auto kernel_event = q.submit([&](handler &h) {
      // the kernel must wait until the DMA transfer is done before launching
      // this will only take affect it we actually performed the DMA above
      h.depends_on(dma_event);

      // the producing kernel
      // NO-FORMAT comments are for clang-format
      h.single_task<KernelID>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        kernel_ptr_type ptr(kernel_ptr);
        for (size_t i = 0; i < count; i++) {
          auto d = *(ptr + i);
          Pipe::write(d);
        }
      });
    });

    return std::make_pair(dma_event, kernel_event);
  }
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Consumer implementation
template <typename Id, typename T, bool use_host_alloc, size_t min_capacity>
class ConsumerImpl : public ProducerConsumerBaseImpl<Id, T, use_host_alloc> {
 private:
  // base implementation alias
  using BaseImpl = ProducerConsumerBaseImpl<Id, T, use_host_alloc>;
  using kernel_ptr_type = typename BaseImpl::kernel_ptr_type;

  // IDs for the pipe and kernel
  class PipeID;
  class KernelID;

  // private constructor so users cannot make an object
  ConsumerImpl(){};

 public:
  // disable copy constructor and operator=
  ConsumerImpl(const ConsumerImpl &) = delete;
  ConsumerImpl &operator=(ConsumerImpl const &) = delete;

  // the pipe to connect to in device code
  using Pipe = sycl::INTEL::pipe<PipeID, T, min_capacity>;

  static std::pair<event, event> Start(queue &q,
                                       size_t count = BaseImpl::count_) {
    // make sure initialized has been called
    BaseImpl::initialized_check();

    // can't produce more data than exists
    if (count > BaseImpl::count_) {
      std::cerr << "ERROR: Start() called with count=" << count
                << " but allocated size is " << BaseImpl::count_ << "\n";
      std::terminate();
    }

    // pick the right pointer to pass to the kernel
    auto kernel_ptr = BaseImpl::get_kernel_ptr();

    // launch the kernel to read the output into device side global memory
    auto kernel_event = q.submit([&](handler &h) {
      // NO-FORMAT comments are for clang-format
      h.single_task<KernelID>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        kernel_ptr_type ptr(kernel_ptr);
        for (size_t i = 0; i < count; i++) {
          auto d = Pipe::read();
          *(ptr + i) = d;
        }
      });
    });

    // if the user wanted to use board memory, copy the data back to the host
    event dma_event;
    if (!use_host_alloc) {
      // launch a task to copy the data back from the device. Use the
      // event.depends_on signal to wait for the kernel to finish first.
      dma_event = q.submit([&](handler &h) {
        h.depends_on(kernel_event);
        h.memcpy(BaseImpl::host_data_, BaseImpl::device_data_,
                 BaseImpl::count_ * sizeof(T));
      });
    }

    return std::make_pair(dma_event, kernel_event);
  }
};
////////////////////////////////////////////////////////////////////////////////

}  // namespace detail

// alias the implementations to face the user
template <typename Id, typename T, bool use_host_alloc, size_t min_capacity = 0>
using Producer = detail::ProducerImpl<Id, T, use_host_alloc, min_capacity>;

template <typename Id, typename T, bool use_host_alloc, size_t min_capacity = 0>
using Consumer = detail::ConsumerImpl<Id, T, use_host_alloc, min_capacity>;

// convenient aliases to get a host or device allocation producer/consumer
template <typename Id, typename T, size_t min_capacity = 0>
using HostConsumer = Consumer<Id, T, true, min_capacity>;

template <typename Id, typename T, size_t min_capacity = 0>
using DeviceConsumer = Consumer<Id, T, false, min_capacity>;

template <typename Id, typename T, size_t min_capacity = 0>
using HostProducer = Producer<Id, T, true, min_capacity>;

template <typename Id, typename T, size_t min_capacity = 0>
using DeviceProducer = Producer<Id, T, false, min_capacity>;

#endif /* __FAKEIOPIPES_HPP__ */
