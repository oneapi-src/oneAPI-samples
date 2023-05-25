//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// Summation of 10M 'one' values
constexpr size_t N = (10 * 1024 * 1024);

constexpr int warm_up_token = -1;

// expected value of sum
int sum_expected = N;

static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

void flush_cache(sycl::queue &q, sycl::buffer<int> &flush_buf,
                 const size_t flush_size) {
  auto ev = q.submit([&](auto &h) {
    sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
  });
  ev.wait_and_throw();
}

int ComputeSerial(std::vector<int> &data,
                  [[maybe_unused]] std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  Timer timer;
  int sum;
  // ComputeSerial main begin
  for (int it = 0; it < iter; it++) {
    sum = 0;
    for (size_t i = 0; i < data_size; ++i) {
      sum += data[i];
    }
  }
  // ComputeSerial main end
  double elapsed = timer.Elapsed() / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeSerial   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeSerial Expected " << sum_expected << " but got "
              << sum << std::endl;
  return sum;
} // end ComputeSerial

int ComputeParallel1(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel1 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
      // ComputeParallel1 main end
    });
    q.wait();
    {
      // ensure limited life-time of host accessor since it blocks the queue
      sycl::host_accessor h_acc(sum_buf);
      sum = h_acc[0];
    }
    // do not measure time of warm-up iteration to exclude JIT compilation time
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel1   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel1 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel1

int ComputeParallel2(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_processing_elements =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  int BATCH = (N + num_processing_elements - 1) / num_processing_elements;
  std::cout << "Num work items = " << num_processing_elements << std::endl;

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_processing_elements);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    // init the acummulator on device
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_processing_elements,
                     [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel2 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_processing_elements, [=](auto index) {
        size_t glob_id = index[0];
        size_t start = glob_id * BATCH;
        size_t end = (glob_id + 1) * BATCH;
        if (end > N)
          end = N;
        int sum = 0;
        for (size_t i = start; i < end; ++i)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });
    // ComputeParallel2 main end
    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_processing_elements; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel2   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel2 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel2

int ComputeTreeReduction1(sycl::queue &q, std::vector<int> &data,
                          std::vector<int> flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int work_group_size = 256;
  int num_work_items = data_size;
  int num_work_groups = num_work_items / work_group_size;

  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping one stage reduction example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  std::cout << "One Stage Reduction with " << num_work_items << std::endl;

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_work_groups);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups,
                     [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeTreeReduction1 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < data_size)
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0)
                         accum_acc[group_id] = scratch[0];
                     });
    });
    // ComputeTreeReduction1 main end
    q.wait();
    {
      sycl::host_accessor h_acc(accum_buf);
      sum = 0;
      for (int i = 0; i < num_work_groups; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeTreeReduction1   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeTreeReduction1 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeTreeReduction

int ComputeTreeReduction2(sycl::queue &q, std::vector<int> &data,
                          std::vector<int> flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int work_group_size = 256;
  int num_work_items1 = data_size;
  int num_work_groups1 = num_work_items1 / work_group_size;
  int num_work_items2 = num_work_groups1;
  int num_work_groups2 = num_work_items2 / work_group_size;

  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping two stage reduction example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  std::cout << "Two Stage Reduction with " << num_work_items1
            << " in stage 1 and " << num_work_items2 << " in stage2"
            << std::endl;

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum1_buf(num_work_groups1);
  sycl::buffer<int> accum2_buf(num_work_groups2);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum1_acc(accum1_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups1,
                     [=](auto index) { accum1_acc[index] = 0; });
    });
    q.submit([&](auto &h) {
      sycl::accessor accum2_acc(accum2_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups2,
                     [=](auto index) { accum2_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeTreeReduction2 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum1_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(num_work_items1, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < data_size)
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0)
                         accum_acc[group_id] = scratch[0];
                     });
    });
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(accum1_buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum2_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(num_work_items2, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < static_cast<size_t>(num_work_items2))
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0)
                         accum_acc[group_id] = scratch[0];
                     });
    });
    // ComputeTreeReduction2 main end
    q.wait();
    {
      sycl::host_accessor h_acc(accum2_buf);
      sum = 0;
      for (int i = 0; i < num_work_groups2; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeTreeReduction2   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeTreeReduction2 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeTreeReduction2

int ComputeParallel3(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_processing_elements =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  int vec_size =
      q.get_device().get_info<sycl::info::device::native_vector_width_int>();
  int num_work_items = num_processing_elements * vec_size;

  std::cout << "Num work items = " << num_work_items << std::endl;
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_work_items);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_items, [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel3 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_items, [=](auto index) {
        size_t glob_id = index[0];
        int sum = 0;
        for (size_t i = glob_id; i < data_size; i += num_work_items)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });
    // ComputeParallel3 main end
    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_work_items; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel3   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel3 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel3

int ComputeParallel4(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  int work_group_size = 256;
  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping ComputeParallel4 example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_processing_elements =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  int vec_size =
      q.get_device().get_info<sycl::info::device::native_vector_width_int>();
  int num_work_items = num_processing_elements * vec_size * work_group_size;
  std::cout << "Num work items = " << num_work_items << std::endl;
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_work_items);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_items, [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel4 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_items, [=](auto index) {
        size_t glob_id = index[0];
        int sum = 0;
        for (size_t i = glob_id; i < data_size; i += num_work_items)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });
    // ComputeParallel4 main end
    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_work_items; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel4   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel4 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel4

int ComputeParallel5(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush.size(), props);
  sycl::buffer<int> sum_buf(&sum, 1, props);
  std::cout << "Compiler built-in reduction Operator" << std::endl;

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(1, [=](auto index) { sum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel5 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      auto sumr = sycl::reduction(sum_buf, h, sycl::plus<>());
      h.parallel_for(sycl::nd_range<1>{data_size, 256}, sumr,
                     [=](sycl::nd_item<1> item, auto &sumr_arg) {
                       int glob_id = item.get_global_id(0);
                       sumr_arg += buf_acc[glob_id];
                     });
    });
    // ComputeParallel5 main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      sum = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel5   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel5 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel5

int ComputeParallel6(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  int work_group_size = 256;
  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping ComputeParallel6 example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  int log2elements_per_block = 13;
  int elements_per_block = (1 << log2elements_per_block); // 8192

  int log2workitems_per_block = 8;
  int workitems_per_block = (1 << log2workitems_per_block); // 256
  int elements_per_work_item = elements_per_block / workitems_per_block;

  int mask = ~(~0 << log2workitems_per_block);
  int num_work_items = data_size / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;
  std::cout << "Num work items = " << num_work_items << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_work_groups);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups,
                     [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel6 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size},
                     [=](sycl::nd_item<1> item) {
                       size_t glob_id = item.get_global_id(0);
                       size_t group_id = item.get_group(0);
                       size_t loc_id = item.get_local_id(0);
                       int offset = ((glob_id >> log2workitems_per_block)
                                     << log2elements_per_block) +
                                    (glob_id & mask);
                       int sum = 0;
                       for (int i = 0; i < elements_per_work_item; ++i)
                         sum +=
                             buf_acc[(i << log2workitems_per_block) + offset];
                       scratch[loc_id] = sum;
                       // Serial Reduction
                       item.barrier(sycl::access::fence_space::local_space);
                       if (loc_id == 0) {
                         int sum = 0;
                         for (int i = 0; i < work_group_size; ++i)
                           sum += scratch[i];
                         accum_acc[group_id] = sum;
                       }
                     });
    });
    // ComputeParallel6 main end
    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_work_groups; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel6   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel6 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel6

int ComputeParallel7(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  int work_group_size = 256;
  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping ComputeParallel7 example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  int log2elements_per_block = 13;
  int elements_per_block = (1 << log2elements_per_block); // 8192

  int log2workitems_per_block = 8;
  int workitems_per_block = (1 << log2workitems_per_block); // 256
  int elements_per_work_item = elements_per_block / workitems_per_block;

  int mask = ~(~0 << log2workitems_per_block);
  int num_work_items = data_size / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;

  std::cout << "Num work items = " << num_work_items << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> accum_buf(num_work_groups);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups,
                     [=](auto index) { accum_acc[index] = 0; });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel7 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size},
                     [=](sycl::nd_item<1> item) {
                       size_t glob_id = item.get_global_id(0);
                       size_t group_id = item.get_group(0);
                       size_t loc_id = item.get_local_id(0);
                       int offset = ((glob_id >> log2workitems_per_block)
                                     << log2elements_per_block) +
                                    (glob_id & mask);
                       int sum = 0;
                       for (int i = 0; i < elements_per_work_item; ++i)
                         sum +=
                             buf_acc[(i << log2workitems_per_block) + offset];
                       scratch[loc_id] = sum;
                       // tree reduction
                       item.barrier(sycl::access::fence_space::local_space);
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (loc_id < static_cast<size_t>(i))
                           scratch[loc_id] += scratch[loc_id + i];
                       }
                       if (loc_id == 0)
                         accum_acc[group_id] = scratch[0];
                     });
    });
    // ComputeParallel7 main end
    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_work_groups; ++i)
        sum += h_acc[i];
      elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel7   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel7 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel7

int ComputeParallel8(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();

  int work_group_size = 512;
  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping ComputeParallel8 example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  int log2elements_per_block = 13;
  int elements_per_block = (1 << log2elements_per_block); // 8192

  int log2workitems_per_block = 8;
  int workitems_per_block = (1 << log2workitems_per_block); // 256
  int elements_per_work_item = elements_per_block / workitems_per_block;

  int mask = ~(~0 << log2workitems_per_block);
  int num_work_items = data_size / elements_per_work_item;

  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush.size(), props);
  sycl::buffer<int> sum_buf(&sum, 1, props);
  std::cout << "Compiler built-in reduction Operator with increased work"
            << std::endl;
  std::cout << "Elements per item = " << elements_per_work_item << std::endl;

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(1, [=](auto index) { sum_acc[index] = 0; });
    });
    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel8 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      auto sumr = sycl::reduction(sum_buf, h, sycl::plus<>());
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size}, sumr,
                     [=](sycl::nd_item<1> item, auto &sumr_arg) {
                       size_t glob_id = item.get_global_id(0);
                       int offset = ((glob_id >> log2workitems_per_block)
                                     << log2elements_per_block) +
                                    (glob_id & mask);
                       int sum = 0;
                       for (int i = 0; i < elements_per_work_item; ++i)
                         sum +=
                             buf_acc[(i << log2workitems_per_block) + offset];
                       sumr_arg += sum;
                     });
    });
    // ComputeParallel8 main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      sum = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel8   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel8 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel8

int ComputeParallel9(sycl::queue &q, std::vector<int> &data,
                     std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum;

  int work_group_size = 512;
  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping ComputeParallel9 example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return 0;
  }
  int log2elements_per_work_item = 6;
  int elements_per_work_item = (1 << log2elements_per_work_item); // 256
  int num_work_items = data_size / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;

  std::cout << "Num work items = " << num_work_items << std::endl;
  std::cout << "Num work groups = " << num_work_groups << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> res_buf(&sum, 1);
  sycl::buffer<sycl::vec<int, 8>> accum_buf(num_work_groups);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; ++i) {
    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_groups, [=](auto index) {
        sycl::vec<int, 8> x{0, 0, 0, 0, 0, 0, 0, 0};
        accum_acc[index] = x;
      });
    });

    flush_cache(q, flush_buf, flush_size);

    Timer timer;
    // ComputeParallel9 main begin
    q.submit([&](auto &h) {
      const sycl::accessor buf_acc(buf, h);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<sycl::vec<int, 8>, 1l> scratch(work_group_size, h);
      h.parallel_for(
          sycl::nd_range<1>{num_work_items, work_group_size},
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            size_t group_id = item.get_group(0);
            size_t loc_id = item.get_local_id(0);
            sycl::sub_group sg = item.get_sub_group();
            sycl::vec<int, 8> sum{0, 0, 0, 0, 0, 0, 0, 0};
            using global_ptr =
                sycl::multi_ptr<int, sycl::access::address_space::global_space>;
            int base = (group_id * work_group_size +
                        sg.get_group_id()[0] * sg.get_local_range()[0]) *
                       elements_per_work_item;
            for (int i = 0; i < elements_per_work_item / 8; ++i)
              sum += sg.load<8>(global_ptr(&buf_acc[base + i * 128]));
            scratch[loc_id] = sum;
            for (int i = work_group_size / 2; i > 0; i >>= 1) {
              item.barrier(sycl::access::fence_space::local_space);
              if (loc_id < static_cast<size_t>(i))
                scratch[loc_id] += scratch[loc_id + i];
            }
            if (loc_id == 0)
              accum_acc[group_id] = scratch[0];
          });
    });
    // ComputeParallel9 main end
    q.wait();
    {
      sycl::host_accessor h_acc(accum_buf);
      sycl::vec<int, 8> res{0, 0, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < num_work_groups; ++i)
        res += h_acc[i];
      sum = 0;
      for (int i = 0; i < 8; ++i)
        sum += res[i];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeParallel9   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else
    std::cout << "ERROR: ComputeParallel9 Expected " << sum_expected
              << " but got " << sum << std::endl;
  return sum;
} // end ComputeParallel9

int main(void) {

  sycl::queue q{sycl::default_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  std::vector<int> data(N, 1);
  std::vector<int> extra(N, 1);

  ComputeSerial(data, extra, 16);
  ComputeParallel1(q, data, extra, 16);
  ComputeParallel2(q, data, extra, 16);
  ComputeTreeReduction1(q, data, extra, 16);
  ComputeTreeReduction2(q, data, extra, 16);
  ComputeParallel3(q, data, extra, 16);
  ComputeParallel4(q, data, extra, 16);
  ComputeParallel5(q, data, extra, 16);
  ComputeParallel6(q, data, extra, 16);
  ComputeParallel7(q, data, extra, 16);
  ComputeParallel8(q, data, extra, 16);
  ComputeParallel9(q, data, extra, 16);
}
