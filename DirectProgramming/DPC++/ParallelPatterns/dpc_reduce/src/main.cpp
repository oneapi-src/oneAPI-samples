//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <mpi.h>
#include <CL/sycl.hpp>
#include <iomanip>  // setprecision library
#include <iostream>

// The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include
// on your development system.
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
constexpr int master = 0;

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
    accessor writeresult(buf,h,write_only);
    h.parallel_for(range<1>{num_steps}, [=](id<1> idx) {
      float x = ((float)idx[0] - 0.5) / (float)num_steps;
      writeresult[idx[0]] = 4.0f / (1.0 + x * x);
    });
  });
  policy.queue().wait();

  // Single task is needed here to make sure
  // data is not written over.
  policy.queue().submit([&](handler& h) {
    accessor a(buf,h);
    h.single_task([=]() {
      for (int i = 1; i < num_steps; i++) a[0] += a[i];
    });
  });
  policy.queue().wait();


  // float mynewresult = buf.get_access<access::mode::read>()[0] / (float)num_steps;
  host_accessor answer(buf,read_only) ; 
  float mynewresult = answer[0]/(float)num_steps; 
  
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
    accessor writeresult(buf, h, write_only); 
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
      accessor my_a(buf,h,read_only);
      accessor my_c(bufc,h,write_only); 
      h.single_task([=]() {
        for (int i = 0 + group_size * j; i < group_size + group_size * j; i++)
          my_c[j] += my_a[i];
      });
    });
  }
  policy.queue().wait();

  host_accessor src(bufc,read_only);

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


// a way to get value_type from both accessors and USM that is needed for transform_init
template <typename Unknown>
struct accessor_traits
{
};

template <typename T, int Dim, sycl::access::mode AccMode, sycl::access::target AccTarget,
          sycl::access::placeholder Placeholder>
struct accessor_traits<sycl::accessor<T, Dim, AccMode, AccTarget, Placeholder>>
{
    using value_type = typename sycl::accessor<T, Dim, AccMode, AccTarget, Placeholder>::value_type;
};

template <typename RawArrayValueType>
struct accessor_traits<RawArrayValueType*>
{
    using value_type = RawArrayValueType;
};

// calculate shift where we should start processing on current item
template <typename NDItemId, typename GlobalIdx, typename SizeNIter, typename SizeN>
SizeN
calc_shift(const NDItemId item_id, const GlobalIdx global_idx, SizeNIter& n_iter, const SizeN n)
{
    auto global_range_size = item_id.get_global_range().size();

    auto start = n_iter * global_idx;
    auto global_shift = global_idx + n_iter * global_range_size;
    if (n_iter > 0 && global_shift > n)
    {
        start += n % global_range_size - global_idx;
    }
    else if (global_shift < n)
    {
        n_iter++;
    }
    return start;
}


template <typename ExecutionPolicy, typename Operation1, typename Operation2>
struct transform_init
{
    Operation1 binary_op;
    Operation2 unary_op;

    template <typename NDItemId, typename GlobalIdx, typename Size, typename AccLocal, typename... Acc>
    void
    operator()(const NDItemId item_id, const GlobalIdx global_idx, Size n, AccLocal& local_mem,
               const Acc&... acc)
    {
        auto local_idx = item_id.get_local_id(0);
        auto global_range_size = item_id.get_global_range().size();
        auto n_iter = n / global_range_size;
        auto start = calc_shift(item_id, global_idx, n_iter, n);
        auto shifted_global_idx = global_idx + start;

        typename accessor_traits<AccLocal>::value_type res;
        if (global_idx < n)
        {
            res = unary_op(shifted_global_idx, acc...);
        }
        // Add neighbour to the current local_mem
        for (decltype(n_iter) i = 1; i < n_iter; ++i)
        {
            res = binary_op(res, unary_op(shifted_global_idx + i, acc...));
        }
        if (global_idx < n)
        {
            local_mem[local_idx] = res;
        }
    }
};


// Reduce on local memory
template <typename ExecutionPolicy, typename BinaryOperation1, typename Tp>
struct reduce
{
    BinaryOperation1 bin_op1;

    template <typename NDItemId, typename GlobalIdx, typename Size, typename AccLocal>
    Tp
    operator()(const NDItemId item_id, const GlobalIdx global_idx, const Size n, AccLocal& local_mem)
    {
        auto local_idx = item_id.get_local_id(0);
        auto group_size = item_id.get_local_range().size();

        auto k = 1;
        do
        {
            item_id.barrier(sycl::access::fence_space::local_space);
            if (local_idx % (2 * k) == 0 && local_idx + k < group_size && global_idx < n &&
                global_idx + k < n)
            {
                local_mem[local_idx] = bin_op1(local_mem[local_idx], local_mem[local_idx + k]);
            }
            k *= 2;
        } while (k < group_size);
        return local_mem[local_idx];
    }
};


// walk through the data
template <typename ExecutionPolicy, typename F>
struct walk_n
{
    F f;

    template <typename ItemId, typename... Ranges>
    auto
    operator()(const ItemId idx, Ranges&&... rngs) -> decltype(f(rngs[idx]...))
    {
        return f(rngs[idx]...);
    }
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
    accessor writeresult(buf,h,write_only);

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

  using Functor = walk_n<Policy, my_no_op>;
  float result;

  // Functor will do nothing for tranform_init and will use plus for reduce.
  // In this example we have done the calculation and filled the buffer above
  // The way transform_init works is that you need to have the value already
  // populated in the buffer.
  auto tf_init = transform_init<Policy, std::plus<float>,
                   Functor>{std::plus<float>(), Functor{my_no_op()}};

  auto combine = std::plus<float>();
  auto brick_reduce = reduce<Policy, std::plus<float>, float>{
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
        accessor access_buf(buf,h);
        accessor temp_acc(temp_buf,h,write_only);
        // Create temporary local buffer
        accessor<float, 1, access::mode::read_write, access::target::local>
            temp_buf_local(range<1>(workgroup_size), h);
        h.parallel_for(nd_range<1>(range<1>(n_groups * workgroup_size),
                                   range<1>(workgroup_size)),
                       [=](nd_item<1> item_id) mutable {
                         auto global_idx = item_id.get_global_id(0);
                         // 1. Initialization (transform part).
                         tf_init(item_id, global_idx, num_steps,
                                 temp_buf_local, access_buf);
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
        accessor temp_acc(temp_buf,h);
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
  
  host_accessor answer(temp_buf,read_only) ; 
  return answer[0]/(float)num_steps; 
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
    accessor writeresult(buf2,h);
    h.parallel_for(range<1>{num_steps},
                   [=](id<1> idx) { writeresult[idx[0]] = (float)idx[0]; });
  });
  policy.queue().wait();

  auto calc_begin = oneapi::dpl::begin(buf2);
  auto calc_end = oneapi::dpl::end(buf2);

  using Functor2 = walk_n<Policy, slice_area>;

  // The buffer has 1...num it at and now we will use that as an input
  // to the slice structue which will calculate the area of each
  // rectangle.
  auto tf_init = transform_init<Policy, std::plus<float>,
                                                 Functor2>{
          std::plus<float>(), Functor2{slice_area(num_steps)}};

  auto combine = std::plus<float>();
  auto brick_reduce = reduce<Policy, std::plus<float>, float>{
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
        accessor access_buf(buf2,h);
        accessor temp_acc(temp_buf,h,write_only);
        // Create temporary local buffer
        accessor<float, 1, access::mode::read_write, access::target::local>
            temp_buf_local(range<1>(workgroup_size), h);
        h.parallel_for(nd_range<1>(range<1>(n_groups * workgroup_size),
                                   range<1>(workgroup_size)),
                       [=](nd_item<1> item_id) mutable {
                         auto global_idx = item_id.get_global_id(0);
                         // 1. Initialization (transform part). Fill local
                         // memory
                         tf_init(item_id, global_idx, num_steps,
                                 temp_buf_local, access_buf);
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
        accessor temp_acc(temp_buf,h);
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
  host_accessor answer(temp_buf,read_only) ; 
  return answer[0]/(float)num_steps; 
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

////////////////////////////////////////////////////////////////////////
//
// Each MPI ranks compute the number Pi partially on target device using DPC++.
// The partial result of number Pi is returned in "results".
//
////////////////////////////////////////////////////////////////////////
void mpi_native(float* results, int rank_num, int num_procs,
                long total_num_steps, queue& q) {
  int num_step_per_rank = total_num_steps / num_procs;
  float dx, dx2;

  dx = 1.0f / (float)total_num_steps;
  dx2 = dx / 2.0f;

  default_selector device_selector;

  // exception handler
  //
  // The exception_list parameter is an iterable list of std::exception_ptr
  // objects. But those pointers are not always directly readable. So, we
  // rethrow the pointer, catch it,  and then we have the exception itself.
  // Note: depending upon the operation there may be several exceptions.
  auto exception_handler = [&](exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "Failure"
                  << "\n";
        std::terminate();
      }
    }
  };

  try {
    // The size of amount of memory that will be given to the buffer.
    range<1> num_items{total_num_steps / size_t(num_procs)};

    // Buffers are used to tell SYCL which data will be shared between the host
    // and the devices.
    buffer<float, 1> results_buf(results,
                                 range<1>(total_num_steps / size_t(num_procs)));

    // Submit takes in a lambda that is passed in a command group handler
    // constructed at runtime.
    q.submit([&](handler& h) {
      // Accessors are used to get access to the memory owned by the buffers.
      accessor results_accessor(results_buf,h,write_only);
      // Each kernel calculates a partial of the number Pi in parallel.
      h.parallel_for(num_items, [=](id<1> k) {
        float x = ((float)rank_num / (float)num_procs) + (float)k * dx + dx2;
        results_accessor[k] = (4.0f * dx) / (1.0f + x * x);
      });
    });
  } catch (...) {
    std::cout << "Failure" << std::endl;
  }
}

// This function uses the DPC++ library call transform reduce.
// It does everything in one library call.
template <typename Policy>
float mpi_dpstd_onestep(int id, int num_procs, long total_num_steps,
                        Policy& policy) {
  int num_step_per_rank = total_num_steps / num_procs;
  float step = 1.0f / (float)total_num_steps;

  float total = std::transform_reduce(
      policy, oneapi::dpl::counting_iterator<int>(1),
      oneapi::dpl::counting_iterator<int>(num_step_per_rank), 0.0f,
      std::plus<float>(), [=](int i) {
        float x = ((float)id / (float)num_procs) + i * step - step / 2;
        return (4.0f / (1.0f + x * x));
      });
  total = total * (float)step;

  return total;
}

int main(int argc, char** argv) {
  int num_steps = 1000000;
  int groups = 10000;
  char machine_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  int id;
  int num_procs;
  float pi;
  queue myQueue{property::queue::in_order()};
  auto policy = oneapi::dpl::execution::make_device_policy(
      queue(default_selector{}, dpc_common::exception_handler));

  // Start MPI.
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "Failed to initialize MPI\n";
    exit(-1);
  }

  // Create the communicator, and retrieve the number of MPI ranks.
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Determine the rank number.
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  // Get the machine name.
  MPI_Get_processor_name(machine_name, &name_len);

  std::cout << "Rank #" << id << " runs on: " << machine_name
            << ", uses device: "
            << myQueue.get_device().get_info<info::device::name>() << "\n";

  if (id == master) {
    printf("Number of steps is %d\n", num_steps);

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
  }

  int num_step_per_rank = num_steps / num_procs;
  float* results_per_rank = new float[num_step_per_rank];

  // Initialize an array to store a partial result per rank.
  for (size_t i = 0; i < num_step_per_rank; i++) results_per_rank[i] = 0.0;

  dpc_common::TimeInterval T6;
  // Calculate the Pi number partially by multiple MPI ranks.
  mpi_native(results_per_rank, id, num_procs, num_steps, myQueue);

  float local_sum = 0.0;

  // Use the DPC++ library call to reduce the array using plus
  buffer<float> calc_values(results_per_rank, num_step_per_rank);
  auto calc_begin2 = dpstd::begin(calc_values);
  auto calc_end2 = dpstd::end(calc_values);

  local_sum =
      std::reduce(policy, calc_begin2, calc_end2, 0.0f, std::plus<float>());
  policy.queue().wait();

  // Master rank performs a reduce operation to get the sum of all partial Pi.
  MPI_Reduce(&local_sum, &pi, 1, MPI_FLOAT, MPI_SUM, master, MPI_COMM_WORLD);

  if (id == master) {
    auto stop6 = T6.Elapsed();

    std::cout << "mpi native:\t\t";
    std::cout << std::setprecision(3) << "PI =" << pi;
    std::cout << " in " << stop6 << " seconds\n";
  }

  delete[] results_per_rank;

  // mpi_dpstd_onestep
  dpc_common::TimeInterval T7;
  local_sum = mpi_dpstd_onestep(id, num_procs, num_steps, policy);
  auto stop7 = T7.Elapsed();

  // Master rank performs a reduce operation to get the sum of all partial Pi.
  MPI_Reduce(&local_sum, &pi, 1, MPI_FLOAT, MPI_SUM, master, MPI_COMM_WORLD);

  if (id == master) {
    auto stop6 = T7.Elapsed();

    std::cout << "mpi transform_reduce:\t";
    std::cout << std::setprecision(3) << "PI =" << pi;
    std::cout << " in " << stop7 << " seconds\n";
    std::cout << "success\n";
  }

  MPI_Finalize();

  return 0;
}
