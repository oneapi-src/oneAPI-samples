//==---- device.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__

#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <map>
#include <vector>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif


namespace dpct {

/// SYCL default exception handler
auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};

class device_info {
public:

  int get_integrated() const { return _integrated; }

  int get_max_clock_frequency() const { return _frequency; }

  int get_max_compute_units() const { return _max_compute_units; }

  void set_name(const char* name) {
    size_t length = strlen(name);
    if (length < 256) {
      std::memcpy(_name, name, length + 1);
    } else {
      std::memcpy(_name, name, 255);
      _name[255] = '\0';
    }
  }

  void set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes) {
    _max_work_item_sizes = max_work_item_sizes;
    for (int i = 0; i < 3; ++i)
      _max_work_item_sizes_i[i] = max_work_item_sizes[i];
  }

  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }

  void set_major_version(int major) { _major = major; }

  void set_minor_version(int minor) { _minor = minor; }

  void set_max_clock_frequency(int frequency) { _frequency = frequency; }

  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }

  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }

  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }

  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }

  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }

  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }

  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++) {
      _max_nd_range_size[i] = max_nd_range_size[i];
      _max_nd_range_size_i[i] = max_nd_range_size[i];
    }
  }

  void set_memory_clock_rate(unsigned int memory_clock_rate) {
    _memory_clock_rate = memory_clock_rate;
  }

  void set_memory_bus_width(unsigned int memory_bus_width) {
    _memory_bus_width = memory_bus_width;
  }

  void
  set_max_register_size_per_work_group(int max_register_size_per_work_group) {
    _max_register_size_per_work_group = max_register_size_per_work_group;
  }

  void set_device_id(uint32_t device_id) {
    _device_id = device_id;
  }

  void set_uuid(std::array<unsigned char, 16> uuid) {
    _uuid = std::move(uuid);
  }

private:
  char _name[256];
  sycl::id<3> _max_work_item_sizes;
  int _max_work_item_sizes_i[3];
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  // Set estimated value 3200000 kHz as default value.
  unsigned int _memory_clock_rate = 3200000;
  // Set estimated value 64 bits as default value.
  unsigned int _memory_bus_width = 64;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  int _max_register_size_per_work_group;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
  int _max_nd_range_size_i[3];
  uint32_t _device_id;
  std::array<unsigned char, 16> _uuid;
};

/// dpct device extension
class device_ext : public sycl::device {
  typedef std::mutex mutex_type;

public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<mutex_type> lock(m_mutex);
    clear_queues();
  }
  device_ext(const sycl::device &base) : sycl::device(base), _ctx(*this) {
    std::lock_guard<mutex_type> lock(m_mutex);
    init_queues();
  }

  int get_major_version() const {
    int major, minor;
    get_version(major, minor);
    return major;
  }

  int get_minor_version() const {
    int major, minor;
    get_version(major, minor);
    return minor;
  }

  int get_max_compute_units() const {
    return get_device_info().get_max_compute_units();
  }

  /// Return the maximum clock frequency of this device in KHz.
  int get_max_clock_frequency() const {
    return get_device_info().get_max_clock_frequency();
  }

  int get_integrated() const { return get_device_info().get_integrated(); }

  void get_device_info(device_info &out) const {
    device_info prop;
    prop.set_name(get_info<sycl::info::device::name>().c_str());

    int major, minor;
    get_version(major, minor);
    prop.set_major_version(major);
    prop.set_minor_version(minor);

    prop.set_max_work_item_sizes(
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION<20220902)
        // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes is an enum class element
        get_info<sycl::info::device::max_work_item_sizes>());
#else
        // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by an int
        get_info<sycl::info::device::max_work_item_sizes<3>>());
#endif
    prop.set_host_unified_memory(
        this->has(sycl::aspect::usm_host_allocations));

    prop.set_max_clock_frequency(
        get_info<sycl::info::device::max_clock_frequency>() * 1000);

    prop.set_max_compute_units(
        get_info<sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(
        get_info<sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(
        get_info<sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<sycl::info::device::local_mem_size>());

#if (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
    if (this->has(sycl::aspect::ext_intel_memory_clock_rate)) {
      unsigned int tmp =
          this->get_info<sycl::ext::intel::info::device::memory_clock_rate>();
      if (tmp != 0)
        prop.set_memory_clock_rate(1000 * tmp);
    }
    if (this->has(sycl::aspect::ext_intel_memory_bus_width)) {
      prop.set_memory_bus_width(
          this->get_info<sycl::ext::intel::info::device::memory_bus_width>());
    }
    if (this->has(sycl::aspect::ext_intel_device_id)) {
      prop.set_device_id(
          this->get_info<sycl::ext::intel::info::device::device_id>());
    }
    if (this->has(sycl::aspect::ext_intel_device_info_uuid)) {
      prop.set_uuid(
          this->get_info<sycl::ext::intel::info::device::uuid>());
    }
#elif defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
memory_bus_width are not supported by the compiler used. \
Use 3200000 kHz as memory_clock_rate default value. \
Use 64 bits as memory_bus_width default value.")
#else
#warning "get_device_info: querying memory_clock_rate and \
memory_bus_width are not supported by the compiler used. \
Use 3200000 kHz as memory_clock_rate default value. \
Use 64 bits as memory_bus_width default value."
#endif

    size_t max_sub_group_size = 1;
    std::vector<size_t> sub_group_sizes =
        get_info<sycl::info::device::sub_group_sizes>();

    for (const auto &sub_group_size : sub_group_sizes) {
      if (max_sub_group_size < sub_group_size)
        max_sub_group_size = sub_group_size;
    }

    prop.set_max_sub_group_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(
        get_info<sycl::info::device::max_work_group_size>());
    int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    prop.set_max_nd_range_size(max_nd_range_size);

    // Estimates max register size per work group, feel free to update the value
    // according to device properties.
    prop.set_max_register_size_per_work_group(65536);

    out = prop;
  }

  device_info get_device_info() const {
    device_info prop;
    get_device_info(prop);
    return prop;
  }

  sycl::queue &in_order_queue() { return *_q_in_order; }

  sycl::queue &default_queue() {
    return in_order_queue();
  }

  void queues_wait_and_throw() {
    std::unique_lock<mutex_type> lock(m_mutex);
    std::vector<std::shared_ptr<sycl::queue>> current_queues(
        _queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      q->wait_and_throw();
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
  }

private:
  void clear_queues() {
    _queues.clear();
    _q_in_order = _q_out_of_order = _saved_queue = nullptr;
  }

  void init_queues() {
    _q_in_order = create_queue_impl(true, sycl::property::queue::in_order());
    _q_out_of_order = create_queue_impl(true);
    _saved_queue = &default_queue();
  }

  /// Caller should acquire resource \p m_mutex before calling this function.
  template <class... Properties>
  sycl::queue *create_queue_impl(bool enable_exception_handler,
                                 Properties... properties) {
    sycl::async_handler eh = {};
    if (enable_exception_handler) {
      eh = exception_handler;
    }
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh,
        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
            sycl::property::queue::enable_profiling(),
#endif
            properties...)));

    return _queues.back().get();
  }

  void get_version(int &major, int &minor) const {
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    std::string ver;
    ver = get_info<sycl::info::device::version>();
    std::string::size_type i = 0;
    while (i < ver.size()) {
      if (isdigit(ver[i]))
        break;
      i++;
    }
    major = std::stoi(&(ver[i]));
    while (i < ver.size()) {
      if (ver[i] == '.')
        break;
      i++;
    }
    i++;
    minor = std::stoi(&(ver[i]));
  }

  sycl::queue *_q_in_order, *_q_out_of_order;
  sycl::queue *_saved_queue;
  sycl::context _ctx;
  std::vector<std::shared_ptr<sycl::queue>> _queues;
  mutable mutex_type m_mutex;
};

static inline unsigned int get_tid() {
#if defined(__linux__)
  return syscall(SYS_gettid);
#elif defined(_WIN64)
  return GetCurrentThreadId();
#else
#error "Only support Windows and Linux."
#endif
}

/// device manager
class dev_mgr {

public:

  device_ext &current_device() {
    unsigned int dev_id=current_device_id();
    check_id(dev_id);
    return *_devs[dev_id];
  }

  device_ext &get_device(unsigned int id) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    check_id(id);
    return *_devs[id];
  }

  unsigned int current_device_id() const {
   std::lock_guard<std::recursive_mutex> lock(m_mutex);
   auto it=_thread2dev_map.find(get_tid());
   if(it != _thread2dev_map.end())
      return it->second;
    return DEFAULT_DEVICE_ID;
  }

  void select_device(unsigned int id) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()]=id;
  }

  unsigned int device_count() { return _devs.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::recursive_mutex m_mutex;
  dev_mgr() {
    sycl::device default_device =
        sycl::device(sycl::default_selector_v);
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<sycl::device> sycl_all_devs =
        sycl::device::get_devices(sycl::info::device_type::all);
    // Collect other devices except for the default device.
    if (default_device.is_cpu())
      _cpu_device = 0;
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
  }

  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }

  std::vector<std::shared_ptr<device_ext>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<unsigned int, unsigned int> _thread2dev_map;
  int _cpu_device = -1;
};

/// Util function to get the default queue of current selected device depends on
/// the USM config. Return the default out-of-ordered queue when USM-none is
/// enabled, otherwise return the default in-ordered queue.
static inline sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}

static inline unsigned int select_device(unsigned int id) {
  dev_mgr::instance().select_device(id);
  return id;
}

} // namespace dpct

#endif // __DPCT_DEVICE_HPP__
