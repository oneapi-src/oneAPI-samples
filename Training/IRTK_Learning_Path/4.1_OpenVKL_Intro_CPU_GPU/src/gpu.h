// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sycl/sycl.hpp>

/* Find an Intel GPU and create a SYCL queue on it. */
inline sycl::queue initSyclQueue()
{
  auto IntelGPUDeviceSelector = [](const sycl::device &device) {
    using namespace sycl::info;
    const std::string deviceName = device.get_info<device::name>();
    bool match                   = device.is_gpu() &&
                 device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
                 device.get_backend() == sycl::backend::ext_oneapi_level_zero;
    return match ? 1 : -1;
  };

  sycl::queue syclQueue(IntelGPUDeviceSelector);

  sycl::context syclContext = syclQueue.get_context();
  sycl::device syclDevice   = syclQueue.get_device();
  std::cout << "Target SYCL device: "
            << syclQueue.get_device().get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  return syclQueue;
}

/* An allocator which allocates memory on a SYCL queue, useful for running on
 * GPU.
 */
template <class T>
struct AllocatorSycl
{
  typedef T value_type;

  AllocatorSycl(sycl::queue &syclQueue) : syclQueue(syclQueue){};

  ~AllocatorSycl() = default;

  template <class U>
  bool operator==(const AllocatorSycl<U> &) const = delete;

  template <class U>
  bool operator!=(const AllocatorSycl<U> &) const = delete;

  template <class U>
  AllocatorSycl(const AllocatorSycl<U> &o)
  {
    syclQueue = o.getSyclQueue();
  }

  T *allocate(const size_t n);
  void deallocate(T *const p, size_t);

  sycl::queue getSyclQueue() const
  {
    return syclQueue;
  };

 private:
  sycl::queue syclQueue;
};

// Inlined definitions //////////////////////////////////////////////////////

template <class T>
T *AllocatorSycl<T>::allocate(const size_t size)
{
  if (size == 0) {
    return nullptr;
  }

  if (size > static_cast<size_t>(-1) / sizeof(T)) {
    throw std::bad_array_new_length();
  }

  const size_t numBytes = size * sizeof(T);

  void *memory = sycl::malloc_shared(numBytes, syclQueue);
  if (!memory) {
    throw std::bad_alloc();
  }

  std::memset(memory, 0, numBytes);

  return reinterpret_cast<T *>(memory);
}

template <class T>
void AllocatorSycl<T>::deallocate(T *const ptr, size_t)
{
  if (!ptr) {
    return;
  }

  sycl::free(ptr, syclQueue);
}
