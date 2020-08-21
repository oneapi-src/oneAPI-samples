//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>  // setprecision library
#include <iostream>
// The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include 
// on your development system.
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// cpu_seq is a simple sequential CPU routine
// that calculates all the slices and then
// does a reduction.
float calc_pi_cpu_seq(int num_steps) {
  float step = 1.0 / (float)num_steps;
  float x;
  float sum = 0.0;
  for (int i = 1; i < num_steps; i++) {
    x = (i - 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  return sum / (float)num_steps;
}

// cpu_tbb is a simple parallel_reduce tbb routine
// that calculates all the slices and then
// uses tbb reduce to combine results.
float calc_pi_cpu_tbb(int num_steps) {
  float step = 1.0 / (float)num_steps;

  auto tbbtotal =
      tbb::parallel_reduce(tbb::blocked_range<int>(1, num_steps), 0.0,
                           [=](tbb::blocked_range<int> r, float running_total) {
                             float y;
                             for (int i = r.begin(); i != r.end(); i++) {
                               y = (i - 0.5) * step;
                               running_total += 4.0 / (1.0 + y * y);
                             }
                             return running_total;
                           },
                           std::plus<float>());
  return tbbtotal / (float)num_steps;
}

// dpstd_native uses a parallel_for to fill
// a buffer with all the slice calculations and
// then uses a single_task to combine all the results
// This is not the highest performing example but shows
// how to do calculations directly in dpc++ with
// mininmal complexity.
template <typename Policy>
float calc_pi_dpstd_native(size_t num_steps, Policy&& policy) {
  float step = 1.0 / (float)num_steps;

  float data[num_steps];

  // Create buffer using host allocated "data" array
  buffer<float, 1> buf{data, range<1>{num_steps}};

  policy.queue().submit([&](handler& h) {
    auto writeresult = buf.get_access<access::mode::write>(h);
    h.parallel_for(range<1>{num_steps}, [=](id<1> idx) {
      float x = ((float)idx[0] - 0.5) / (float)num_steps;
      writeresult[idx[0]] = 4.0f / (1.0 + x * x);
    });
  });
  policy.queue().wait();

  // Single task is needed here to make sure
  // data is not written over.
  policy.queue().submit([&](handler& h) {
    auto a = buf.get_access<access::mode::read_write>(h);
    h.single_task([=]() {
      for (int i = 1; i < num_steps; i++) a[0] += a[i];
    });
  });
  policy.queue().wait();

  float mynewresult =
      buf.get_access<access::mode::read>()[0] / (float)num_steps;
  return mynewresult;
}

// This option uses a parallel for to fill the array, and then use a single
// task to reduce into groups and then use cpu for final reduction.
template <typename Policy>
float calc_pi_dpstd_native2(size_t num_steps, Policy&& policy, int group_size) {
  float step = 1.0 / (float)num_steps;

  float data[num_steps];
  float myresult = 0.0;

  // Create buffer using host allocated "data" array
  buffer<float, 1> buf{data, range<1>{num_steps}};

  // fill buffer with calculations
  policy.queue().submit([&](handler& h) {
    auto writeresult = buf.get_access<access::mode::write>(h);
    h.parallel_for(range<1>{num_steps}, [=](id<1> idx) {
      float x = ((float)idx[0] - 0.5) / (float)num_steps;
      writeresult[idx[0]] = 4.0f / (1.0 + x * x);
    });
  });
  policy.queue().wait();

  size_t num_groups = num_steps / group_size;
  float c[num_groups];
  // create a number of groups and do a local reduction
  // within these groups using single_task.  Store each
  // result within the output of bufc
  for (int i = 0; i < num_groups; i++) c[i] = 0;
  buffer<float, 1> bufc{c, range<1>{num_groups}};
  for (int j = 0; j < num_groups; j++) {
    policy.queue().submit([&](handler& h) {
      auto my_a = buf.get_access<access::mode::read>(h);
      auto my_c = bufc.get_access<access::mode::write>(h);
      h.single_task([=]() {
        for (int i = 0 + group_size * j; i < group_size + group_size * j; i++)
          my_c[j] += my_a[i];
      });
    });
  }
  policy.queue().wait();

  auto src = bufc.get_access<access::mode::read>();

  // Sum up results on CPU
  float mynewresult = 0.0;
  for (int i = 0; i < num_groups; i++) mynewresult += src[i];

  return mynewresult / (float)num_steps;
}

// Function operator used as transform operation in transform-reduce operations
// implemented below.
struct my_no_op {
  template <typename Tp>
  Tp&& operator()(Tp&& a) const {
    return std::forward<Tp>(a);
  }
};

// Structure slice area performs the calculations for
// each rectangle that will be summed up.
struct slice_area {
  int num;
  slice_area(int num_steps) { num = num_steps; }

  template <typename T>
  float operator()(T&& i) {
    float x = ((float)i - 0.5) / (float)num;
    return 4.0f / (1.0f + (x * x));
  };
};

// This option uses a parallel for to fill the buffer and then
// uses a tranform_init with plus/no_op and then
// a local reduction then global reduction.
template <typename Policy>
float calc_pi_dpstd_native3(size_t num_steps, int groups, Policy&& policy) {
  float data[num_steps];

  // Create buffer using host allocated "data" array
  buffer<float, 1> buf{data, range<1>{num_steps}};

  // fill the buffer with the calculation using parallel for
  policy.queue().submit([&](handler& h) {
    auto writeresult = buf.get_access<access::mode::write>(h);
    h.parallel_for(range<1>{num_steps}, [=](id<1> idx) {
      float x = (float)idx[0] / (float)num_steps;
      writeresult[idx[0]] = 4.0f / (1.0f + x * x);
    });
  });
  policy.queue().wait();

  // Calc_begin and calc_end are iterators pointing to
  // beginning and end of the buffer
  auto calc_begin = oneapi::dpl::begin(buf);
  auto calc_end = oneapi::dpl::end(buf);

  using Functor = oneapi::dpl::unseq_backend::walk_n<Policy, my_no_op>;
  float result;

  // Functor will do nothing for tranform_init and will use plus for reduce.
  // In this example we have done the calculation and filled the buffer above
  // The way transform_init works is that you need to have the value already
  // populated in the buffer.
  auto tf_init =
      oneapi::dpl::unseq_backend::transform_init<Policy, std::plus<float>,
                                                 Functor>{std::plus<float>(),
                                                          Functor{my_no_op()}};

  auto combine = std::plus<float>();
  auto brick_reduce =
      oneapi::dpl::unseq_backend::reduce<Policy, std::plus<float>, float>{
          std::plus<float>()};
  auto workgroup_size =
      policy.queue()
          .get_device()
          .template get_info<info::device::max_work_group_size>();
  auto max_comp_u = policy.queue()
                        .get_device()
                        .template get_info<info::device::max_compute_units>();
  auto n_groups = (num_steps - 1) / workgroup_size + 1;
  n_groups =
      std::min(decltype(n_groups)(max_comp_u),
               n_groups);  // make groups max number of compute units or less

  // 0. Create temporary global buffer to store temporary value
  auto temp_buf = buffer<float, 1>(range<1>(n_groups));
  // 1. Reduce over each work_group
  auto local_reduce_event =
      policy.queue().submit([&buf, &temp_buf, &brick_reduce, &tf_init,
                             num_steps, n_groups, workgroup_size](handler& h) {
        auto access_buf = buf.template get_access<access::mode::read_write>(h);
        auto temp_acc =
            temp_buf.template get_access<access::mode::discard_write>(h);
        // Create temporary local buffer
        accessor<float, 1, access::mode::read_write, access::target::local>
            temp_buf_local(range<1>(workgroup_size), h);
        h.parallel_for(nd_range<1>(range<1>(n_groups * workgroup_size),
                                   range<1>(workgroup_size)),
                       [=](nd_item<1> item_id) mutable {
                         auto global_idx = item_id.get_global_id(0);
                         // 1. Initialization (transform part).
                         tf_init(item_id, global_idx, access_buf, num_steps,
                                 temp_buf_local);
                         // 2. Reduce within work group
                         float local_result = brick_reduce(
                             item_id, global_idx, num_steps, temp_buf_local);
                         if (item_id.get_local_id(0) == 0) {
                           temp_acc[item_id.get_group(0)] = local_result;
                         }
                       });
      });

  // 2. global reduction
  auto reduce_event = local_reduce_event;
  if (n_groups > 1) {
    auto countby2 = decltype(n_groups)(1);
    do {
      reduce_event = policy.queue().submit([&reduce_event, &temp_buf, &combine,
                                            countby2, n_groups](handler& h) {
        h.depends_on(reduce_event);
        auto temp_acc =
            temp_buf.template get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(n_groups), [=](item<1> item_id) mutable {
          auto global_idx = item_id.get_linear_id();

          if (global_idx % (2 * countby2) == 0 &&
              global_idx + countby2 < n_groups) {
            temp_acc[global_idx] =
                combine(temp_acc[global_idx], temp_acc[global_idx + countby2]);
          }
        });
      });
      countby2 *= 2;
    } while (countby2 < n_groups);
  }

  float answer = temp_buf.template get_access<access::mode::read>()[0];
  result = answer / (float)num_steps;
  return result;
}

// dpstd_native4 fills a buffer with number 1...num_steps and then
// calls transform_init to calculate the slices and then
// does a reduction in two steps - global and then local.
template <typename Policy>
float calc_pi_dpstd_native4(size_t num_steps, int groups, Policy&& policy) {
  std::vector<float> data(num_steps);
  float result = 0.0;

  buffer<float, 1> buf2{data.data(), range<1>{num_steps}};

  // fill buffer with 1...num_steps
  policy.queue().submit([&](handler& h) {
    auto writeresult = buf2.get_access<access::mode::write>(h);
    h.parallel_for(range<1>{num_steps},
                   [=](id<1> idx) { writeresult[idx[0]] = (float)idx[0]; });
  });
  policy.queue().wait();

  auto calc_begin = oneapi::dpl::begin(buf2);
  auto calc_end = oneapi::dpl::end(buf2);

  using Functor2 = oneapi::dpl::unseq_backend::walk_n<Policy, slice_area>;

  // The buffer has 1...num it at and now we will use that as an input
  // to the slice structue which will calculate the area of each
  // rectangle.
  auto tf_init =
      oneapi::dpl::unseq_backend::transform_init<Policy, std::plus<float>,
                                                 Functor2>{
          std::plus<float>(), Functor2{slice_area(num_steps)}};

  auto combine = std::plus<float>();
  auto brick_reduce =
      oneapi::dpl::unseq_backend::reduce<Policy, std::plus<float>, float>{
          std::plus<float>()};

  // get workgroup_size from the device
  auto workgroup_size =
      policy.queue()
          .get_device()
          .template get_info<info::device::max_work_group_size>();

  // get number of compute units from device.
  auto max_comp_u = policy.queue()
                        .get_device()
                        .template get_info<info::device::max_compute_units>();

  auto n_groups = (num_steps - 1) / workgroup_size + 1;

  // use the smaller of the number of workgroups device has or the
  // number of steps/workgroups
  n_groups = std::min(decltype(n_groups)(max_comp_u), n_groups);

  // Create temporary global buffer to store temporary value
  auto temp_buf = buffer<float, 1>(range<1>(n_groups));

  // Reduce over each work_group
  auto local_reduce_event =
      policy.queue().submit([&buf2, &temp_buf, &brick_reduce, &tf_init,
                             num_steps, n_groups, workgroup_size](handler& h) {
        // grab access to the previous input
        auto access_buf = buf2.template get_access<access::mode::read_write>(h);
        auto temp_acc =
            temp_buf.template get_access<access::mode::discard_write>(h);
        // Create temporary local buffer
        accessor<float, 1, access::mode::read_write, access::target::local>
            temp_buf_local(range<1>(workgroup_size), h);
        h.parallel_for(nd_range<1>(range<1>(n_groups * workgroup_size),
                                   range<1>(workgroup_size)),
                       [=](nd_item<1> item_id) mutable {
                         auto global_idx = item_id.get_global_id(0);
                         // 1. Initialization (transform part). Fill local
                         // memory
                         tf_init(item_id, global_idx, access_buf, num_steps,
                                 temp_buf_local);
                         // 2. Reduce within work group
                         float local_result = brick_reduce(
                             item_id, global_idx, num_steps, temp_buf_local);
                         if (item_id.get_local_id(0) == 0) {
                           temp_acc[item_id.get_group(0)] = local_result;
                         }
                       });
      });

  // global reduction
  auto reduce_event = local_reduce_event;
  if (n_groups > 1) {
    auto countby2 = decltype(n_groups)(1);
    do {
      reduce_event = policy.queue().submit([&reduce_event, &temp_buf, &combine,
                                            countby2, n_groups](handler& h) {
        h.depends_on(reduce_event);
        auto temp_acc =
            temp_buf.template get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(n_groups), [=](item<1> item_id) mutable {
          auto global_idx = item_id.get_linear_id();

          if (global_idx % (2 * countby2) == 0 &&
              global_idx + countby2 < n_groups) {
            temp_acc[global_idx] =
                combine(temp_acc[global_idx], temp_acc[global_idx + countby2]);
          }
        });
      });
      countby2 *= 2;
    } while (countby2 < n_groups);
  }
  float answer = temp_buf.template get_access<access::mode::read_write>()[0];
  result = answer / (float)num_steps;

  return result;
}

// This function shows the use of two different DPC++ library calls.
// The first is a transform calls which will fill a buff with the
// calculations of each small rectangle.   The second call is the reduce
// call which sums up the results of all the elements in the buffer.
template <typename Policy>
float calc_pi_dpstd_two_steps_lib(int num_steps, Policy&& policy) {
  float step = 1.0 / (float)num_steps;

  buffer<float> calc_values{num_steps};
  auto calc_begin2 = oneapi::dpl::begin(calc_values);
  auto calc_end2 = oneapi::dpl::end(calc_values);

  // use DPC++ library call transform to fill the buffer with
  // the area calculations for each rectangle.
  std::transform(policy, oneapi::dpl::counting_iterator<int>(1),
                 oneapi::dpl::counting_iterator<int>(num_steps), calc_begin2,
                 [=](int i) {
                   float x = (((float)i - 0.5f) / (float)(num_steps));
                   return (4.0f / (1.0f + x * x));
                 });

  policy.queue().wait();

  // use the DPC++ library call to reduce the array using plus
  float result =
      std::reduce(policy, calc_begin2, calc_end2, 0.0f, std::plus<float>());
  policy.queue().wait();

  result = result / (float)num_steps;

  return result;
}

// This function uses the DPC++ library call
// transform reduce.  It does everything in one library
// call.
template <typename Policy>
float calc_pi_dpstd_onestep(int num_steps, Policy& policy) {
  float step = 1.0f / (float)num_steps;

  float total = std::transform_reduce(
      policy, oneapi::dpl::counting_iterator<int>(1),
      oneapi::dpl::counting_iterator<int>(num_steps), 0.0f, std::plus<float>(),
      [=](int i) {
        float x = (float)(((float)i - 0.5f) / (float(num_steps)));
        return (4.0f / (1.0f + x * x));
      });
  total = total * (float)step;

  return total;
}

int main(int argc, char** argv) {
  int num_steps = 1000000;
  printf("Number of steps is %d\n", num_steps);
  int groups = 10000;

  float pi;
  queue myQueue{property::queue::in_order()};
  auto policy = oneapi::dpl::execution::make_device_policy(
      queue(default_selector{}, dpc_common::exception_handler));

  // Since we are using JIT compiler for samples,
  // we need to run each step once to allow for compile
  // to occur before we time execution of function.
  pi = calc_pi_dpstd_native(num_steps, policy);
  pi = calc_pi_dpstd_native2(num_steps, policy, groups);
  pi = calc_pi_dpstd_native3(num_steps, groups, policy);
  pi = calc_pi_dpstd_native4(num_steps, groups, policy);

  pi = calc_pi_dpstd_two_steps_lib(num_steps, policy);
  pi = calc_pi_dpstd_onestep(num_steps, policy);

  dpc_common::TimeInterval T;
  pi = calc_pi_cpu_seq(num_steps);
  auto stop = T.Elapsed();
  std::cout << "Cpu Seq calc: \t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop << " seconds\n";

  dpc_common::TimeInterval T2;
  pi = calc_pi_cpu_tbb(num_steps);
  auto stop2 = T2.Elapsed();
  std::cout << "Cpu TBB  calc: \t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop2 << " seconds\n";

  dpc_common::TimeInterval T3;
  pi = calc_pi_dpstd_native(num_steps, policy);
  auto stop3 = T3.Elapsed();
  std::cout << "dpstd native:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop3 << " seconds\n";

  dpc_common::TimeInterval T3a;
  pi = calc_pi_dpstd_native2(num_steps, policy, groups);
  auto stop3a = T3a.Elapsed();
  std::cout << "dpstd native2:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop3a << " seconds\n";

  dpc_common::TimeInterval T3b;
  pi = calc_pi_dpstd_native3(num_steps, groups, policy);
  auto stop3b = T3b.Elapsed();
  std::cout << "dpstd native3:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop3b << " seconds\n";

  dpc_common::TimeInterval T3c;
  pi = calc_pi_dpstd_native4(num_steps, groups, policy);
  auto stop3c = T3c.Elapsed();
  std::cout << "dpstd native4:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop3c << " seconds\n";

  dpc_common::TimeInterval T4;
  pi = calc_pi_dpstd_two_steps_lib(num_steps, policy);
  auto stop4 = T4.Elapsed();
  std::cout << "dpstd two steps:\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop4 << " seconds\n";

  dpc_common::TimeInterval T5;
  pi = calc_pi_dpstd_onestep(num_steps, policy);
  auto stop5 = T5.Elapsed();
  std::cout << "dpstd transform_reduce: ";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop5 << " seconds\n";

  std::cout << "success\n";
  return 0;
}
