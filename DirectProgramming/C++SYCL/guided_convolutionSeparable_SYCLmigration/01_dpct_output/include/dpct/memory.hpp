//==---- memory.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MEMORY_HPP__
#define __DPCT_MEMORY_HPP__

#include <sycl/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <map>
#include <utility>
#include <thread>
#include <type_traits>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

#include "device.hpp"

namespace dpct {

enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};

enum memory_region {
  global = 0,  // device global memory
  constant,    // device constant memory
  local,       // device local memory
  shared,      // memory which can be accessed by host and device
};

typedef uint8_t byte_t;

/// Buffer type to be used in Memory Management runtime.
typedef sycl::buffer<byte_t> buffer_t;

/// Pitched 2D/3D memory data.
class pitched_data {
public:
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : _data(data), _pitch(pitch), _x(x), _y(y) {}

  void *get_data_ptr() { return _data; }

  size_t get_pitch() { return _pitch; }

  size_t get_y() { return _y; }

private:
  void *_data;
  size_t _pitch, _x, _y;
};

namespace detail {

class mem_mgr {
  mem_mgr() {
    // Reserved address space, no real memory allocation happens here.
#if defined(__linux__)
    mapped_address_space =
        (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
    mapped_address_space = (byte_t *)VirtualAlloc(
        NULL,               // NULL specified as the base address parameter
        mapped_region_size, // Size of allocation
        MEM_RESERVE,        // Allocate reserved pages
        PAGE_NOACCESS);     // Protection = no access
#else
#error "Only support Windows and Linux."
#endif
    next_free = mapped_address_space;
  };

public:
  using buffer_id_t = int;

  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };

  ~mem_mgr() {
#if defined(__linux__)
    munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
    VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
  };

  mem_mgr(const mem_mgr &) = delete;
  mem_mgr &operator=(const mem_mgr &) = delete;
  mem_mgr(mem_mgr &&) = delete;
  mem_mgr &operator=(mem_mgr &&) = delete;

  /// Returns the instance of memory manager singleton.
  static mem_mgr &instance() {
    static mem_mgr m;
    return m;
  }

private:
  std::map<byte_t *, allocation> m_map;
  mutable std::mutex m_mutex;
  byte_t *mapped_address_space;
  byte_t *next_free;
  const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
  const size_t alignment = 256;
  /// This padding may be defined to some positive value to debug
  /// out of bound accesses.
  const size_t extra_padding = 0;

};

template <class T, memory_region Memory, size_t Dimension> class accessor;
template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr sycl::access::target target =
      (Memory == constant) ? sycl::access::target::constant_buffer
                           : sycl::access::target::device;
  static constexpr sycl::access_mode mode =
      (Memory == constant) ? sycl::access_mode::read
                           : sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional<Memory == constant, const T, T>::type;
  using value_t = typename std::remove_cv<T>::type;
  template <size_t Dimension = 1>
  using accessor_t = typename std::conditional<
      Memory == local, sycl::local_accessor<value_t, Dimension>,
      sycl::accessor<T, Dimension, mode, target>>::type;
  using pointer_t = T *;
};

static inline void *dpct_malloc(size_t size, sycl::queue &q) {
  return sycl::malloc_device(size, q.get_device(), q.get_context());
}

enum class pointer_access_attribute {
  host_only = 0,
  device_only,
  host_device,
  end
};

static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                      const void *ptr) {
  switch (sycl::get_pointer_type(ptr, q.get_context())) {
  case sycl::usm::alloc::unknown:
    return pointer_access_attribute::host_only;
  case sycl::usm::alloc::device:
    return pointer_access_attribute::device_only;
  case sycl::usm::alloc::shared:
  case sycl::usm::alloc::host:
    return pointer_access_attribute::host_device;
  }
}

static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                             const void *from_ptr,
                                             memcpy_direction dir) {
  switch (dir) {
  case memcpy_direction::host_to_host:
  case memcpy_direction::host_to_device:
  case memcpy_direction::device_to_host:
  case memcpy_direction::device_to_device:
    return dir;
  case memcpy_direction::automatic: {
    // table[to_attribute][from_attribute]
    static const memcpy_direction
        direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                       [static_cast<unsigned>(pointer_access_attribute::end)] =
                           {{memcpy_direction::host_to_host,
                             memcpy_direction::device_to_host,
                             memcpy_direction::host_to_host},
                            {memcpy_direction::host_to_device,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device},
                            {memcpy_direction::host_to_host,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device}};
    return direction_table[static_cast<unsigned>(get_pointer_attribute(
        q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
}

static sycl::event
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  if (!size)
    return sycl::event{};
  return q.memcpy(to_ptr, from_ptr, size, dep_events);
}

// Get actual copy range and make sure it will not exceed range.
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
  return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}

static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch) {
  return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id,
            sycl::range<3> size, memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  // RAII for host pointer
  class host_buffer {
    void *_buf;
    size_t _size;
    sycl::queue &_q;
    const std::vector<sycl::event> &_deps; // free operation depends

  public:
    host_buffer(size_t size, sycl::queue &q,
                const std::vector<sycl::event> &deps)
        : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
    void *get_ptr() const { return _buf; }
    size_t get_size() const { return _size; }
    ~host_buffer() {
      if (_buf) {
        _q.submit([&](sycl::handler &cgh) {
          cgh.depends_on(_deps);
          cgh.host_task([buf = _buf] { std::free(buf); });
        });
      }
    }
  };
  std::vector<sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0),
         from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface =
      (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr +
      get_offset(from_id, from_slice, from_range.get(0));

  if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
    return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                        direction, dep_events)};
  }
  direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
  size_t size_slice = size.get(1) * size.get(0);
  switch (direction) {
  case host_to_host:
    for (size_t z = 0; z < size.get(2); ++z) {
      unsigned char *to_ptr = to_surface;
      const unsigned char *from_ptr = from_surface;
      if (to_range.get(0) == from_range.get(0) &&
          to_range.get(0) == size.get(0)) {
        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                         direction, dep_events));
      } else {
        for (size_t y = 0; y < size.get(1); ++y) {
          event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                           direction, dep_events));
          to_ptr += to_range.get(0);
          from_ptr += from_range.get(0);
        }
      }
      to_surface += to_slice;
      from_surface += from_slice;
    }
    break;
  case host_to_device: {
    host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                    event_list);
    std::vector<sycl::event> host_events;
    if (to_slice == size_slice) {
      // Copy host data to a temp host buffer with the shape of target.
      host_events =
          dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                      sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                      host_to_host, dep_events);
    } else {
      // Copy host data to a temp host buffer with the shape of target.
      host_events = dpct_memcpy(
          q, buf.get_ptr(), from_surface, to_range, from_range,
          sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
          // If has padding data, not sure whether it is useless. So fill temp
          // buffer with it.
          std::vector<sycl::event>{
              dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                          device_to_host, dep_events)});
    }
    // Copy from temp host buffer to device with only one submit.
    event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                     buf.get_size(), host_to_device,
                                     host_events));
    break;
  }
  case device_to_host: {
    host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                    event_list);
    // Copy from host temp buffer to host target with reshaping.
    event_list = dpct_memcpy(
        q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
        sycl::id<3>(0, 0, 0), size, host_to_host,
        // Copy from device to temp host buffer with only one submit.
        std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                 buf.get_size(),
                                                 device_to_host, dep_events)});
    break;
  }
  case device_to_device:
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.parallel_for<class dpct_memcpy_3d_detail>(
          size,
          [=](sycl::id<3> id) {
            to_surface[get_offset(id, to_slice, to_range.get(0))] =
                from_surface[get_offset(id, from_slice, from_range.get(0))];
          });
    }));
  break;
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
  return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
            pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                     sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                     sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                     size, direction);
}

/// memcpy 2D matrix with pitch.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            size_t to_pitch, size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                     sycl::range<3>(from_pitch, y, 1),
                     sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                     sycl::range<3>(x, y, 1), direction);
}

} // namespace detail

/// free
/// \param ptr Point to free.
/// \param q Queue to execute the free task.
/// \returns no return value.
static inline void dpct_free(void *ptr,
                             sycl::queue &q = get_default_queue()) {
  if (ptr) {
    sycl::free(ptr, q.get_context());
  }
}

/// Synchronously copies \p size bytes from the address specified by \p from_ptr
/// to the address specified by \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device,
/// \a device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                        memcpy_direction direction = automatic,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction).wait();
}

/// Synchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(void *to_ptr, size_t to_pitch,
                               const void *from_ptr, size_t from_pitch,
                               size_t x, size_t y,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch,
                                            from_pitch, x, y, direction));
}

/// Synchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specified by \p size.
/// The value of \p direction is used to set the copy direction, it can be \a
/// host_to_host, \a host_to_device, \a device_to_host, \a device_to_device or
/// \a automatic. The function will return after the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(pitched_data to, sycl::id<3> to_pos,
                               pitched_data from, sycl::id<3> from_pos,
                               sycl::range<3> size,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(
      detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction));
}

namespace detail {

/// Device variable with address space of shared, global or constant.
template <class T, memory_region Memory, size_t Dimension>
class device_memory {
public:

  using value_t = typename detail::memory_traits<Memory, T>::value_t;

  device_memory() : device_memory(sycl::range<Dimension>(1)) {}

  /// Constructor of 1-D array with initializer list
  device_memory(
      const sycl::range<Dimension> &in_range,
      std::initializer_list<value_t> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
  }

  /// Constructor of 2-D array with initializer list
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range[0]);
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    auto tmp_data = _host_ptr;
    for (auto sub_list : init_list) {
      assert(sub_list.size() <= in_range[1]);
      std::memcpy(tmp_data, sub_list.begin(), sub_list.size() * sizeof(T));
      tmp_data += in_range[1];
    }
  }

  /// Constructor with range
  device_memory(const sycl::range<Dimension> &range_in)
      : _size(range_in.size() * sizeof(T)), _range(range_in), _reference(false),
        _host_ptr(nullptr), _device_ptr(nullptr) {
    static_assert(
        (Memory == global) || (Memory == constant) || (Memory == shared),
        "device memory region should be global, constant or shared");
    // Make sure that singleton class mem_mgr and dev_mgr will destruct later
    // than this.
    detail::mem_mgr::instance();
    dev_mgr::instance();
  }

  /// Constructor with range
  template <class... Args>
  device_memory(Args... Arguments)
      : device_memory(sycl::range<Dimension>(Arguments...)) {}

  ~device_memory() {
    if (_device_ptr && !_reference)
      dpct_free(_device_ptr);
    if (_host_ptr)
      std::free(_host_ptr);
  }

  /// Allocate memory with default queue, and init memory if has initial value.
  void init() {
    init(dpct::get_default_queue());
  }

  /// Allocate memory with specified queue, and init memory if has initial value.
  void init(sycl::queue &q) {
    if (_device_ptr)
      return;
    if (!_size)
      return;
    allocate_device(q);
    if (_host_ptr)
      detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size, host_to_device);
  }

  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr() {
    return get_ptr(get_default_queue());
  }

  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr(sycl::queue &q) {
    init(q);
    return _device_ptr;
  }

  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
    init();
    return _device_ptr[index];
  }

private:
  device_memory(value_t *memory_ptr, size_t size)
      : _size(size), _range(size / sizeof(T)), _reference(true),
        _device_ptr(memory_ptr) {}

  void allocate_device(sycl::queue &q) {
    if (Memory == shared) {
      _device_ptr = (value_t *)sycl::malloc_shared(
          _size, q.get_device(), q.get_context());
      return;
    }
    _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
  }

  size_t _size;
  sycl::range<Dimension> _range;
  bool _reference;
  value_t *_host_ptr;
  value_t *_device_ptr;
};
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;

  /// Constructor with initial value.
  device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

  /// Default constructor
  device_memory() : base(1) {}

};

} // namespace detail

template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;

} // namespace dpct

#endif // __DPCT_MEMORY_HPP__
