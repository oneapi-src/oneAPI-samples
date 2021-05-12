#ifndef __STREAMING_WITHOUT_API_HPP__
#define __STREAMING_WITHOUT_API_HPP__

#include <algorithm>
#include <numeric>
#include <atomic>
#include <queue>
#include <thread>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "common.hpp"

using namespace sycl;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
// forward declare functions
template<typename T>
void DoOneIteration(queue& q, size_t buffers, size_t buffer_count, size_t reps,
                    size_t iterations, size_t threads,
                    T *in_stream, T *out_stream,
                    std::vector<high_resolution_clock::time_point>& time_in,
                    std::vector<high_resolution_clock::time_point>& time_out,
                    high_resolution_clock::time_point& start,
                    high_resolution_clock::time_point& end);

template<typename T>
bool DoWork(queue& q, size_t buffers, size_t buffer_count, size_t reps,
            size_t iterations, size_t threads);

template<typename T>
void ProducerThread(T* in_stream, size_t buffer_count, size_t reps, int threads,
                    std::atomic<bool>& data_valid, T*& out_ptr);

template<typename T>
void KernelThread(queue& q, size_t buffers, size_t buffer_count, size_t reps,
                  int threads,
                  std::vector<T*>& in_buf, std::vector<T*>& out_buf,
                  std::atomic<bool>& produce_data_valid, T*& in_ptr,
                  T* out_stream,
                  std::vector<high_resolution_clock::time_point>& time_in,
                  std::vector<high_resolution_clock::time_point>& time_out);

template<typename T>
void DoOneIteration(queue& q, size_t buffers, size_t buffer_count, size_t reps,
                    size_t iterations, size_t threads,
                    T *in_stream, T *out_stream,
                    std::vector<high_resolution_clock::time_point>& time_in,
                    std::vector<high_resolution_clock::time_point>& time_out,
                    high_resolution_clock::time_point& start,
                    high_resolution_clock::time_point& end);

template<typename T>
event SubmitKernel(queue &q, T *in_ptr, size_t count, T *out_ptr);
////////////////////////////////////////////////////////////////////////////////

//
// This function contains the logic to perform the buffered streaming.
// It performs a single iteration of the design. This function will be called
// multiple times to improve the accuracy of the performance measurement.
//
template<typename T>
void DoOneIteration(queue& q, size_t buffers, size_t buffer_count, size_t reps,
                    size_t iterations, size_t threads,
                    T *in_stream, T *out_stream,
                    std::vector<high_resolution_clock::time_point>& time_in,
                    std::vector<high_resolution_clock::time_point>& time_out,
                    high_resolution_clock::time_point& start,
                    high_resolution_clock::time_point& end) {
  // the Producer and Consumer get half of the total threads, each.
  // std::max() guards against the case where there is only 1 thread.
  size_t half_threads = std::max(size_t(1), threads/2);

  // allocate space for the USM buffers
  std::vector<T*> in(buffers);
  std::vector<T*> out(buffers);
  for (size_t i = 0; i < buffers; i++) {
    if ((in[i] = malloc_host<T>(buffer_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in[" << i << "]'\n";
      std::terminate();
    }
    if ((out[i] = malloc_host<T>(buffer_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out[" << i << "]'\n";
      std::terminate();
    }
  }

  // inter-thread communication variables
  T* produce_ptr(in[0]);
  std::atomic<bool> produce_data_valid(false);

  start = high_resolution_clock::now();

  // start the Producer in a new thread (starts in the ProducerThread<T>
  // function)
  std::thread producer_thread(ProducerThread<T>,
                              in_stream, buffer_count, reps,
                              half_threads,
                              std::ref(produce_data_valid),
                              std::ref(produce_ptr));

  // run the kernel in this thread
  KernelThread<T>(q, buffers, buffer_count, reps, half_threads,
                  in, out, produce_data_valid, produce_ptr,
                  out_stream, time_in, time_out);

  // wait for producer to finish
  producer_thread.join();

  end = high_resolution_clock::now();

  // free the USM buffers
  for (size_t i = 0; i < buffers; i++) {
    sycl::free(in[i], q);
    sycl::free(out[i], q);
  }
}

//
// The top level function for doing work with the design. It deals with timing,
// checking errors, and running multiple iterations to improve performance
// accuracy.
//
template<typename T>
bool DoWork(queue& q, size_t buffers, size_t buffer_count, size_t reps,
            size_t iterations, size_t threads) {
  // track how many errors we detect in the output
  size_t total_errors = 0;

  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // create input and output streams of data for ALL the data to be processed
  size_t total_count = buffer_count * reps;
  std::vector<T> in_stream(total_count);
  std::vector<T> out_stream(total_count);
  
  // generate random input data
  std::generate_n(in_stream.begin(), total_count, [] { return rand() % 100; });

  // track the input and output times for each repetition (latency)
  std::vector<high_resolution_clock::time_point> time_in(reps);
  std::vector<high_resolution_clock::time_point> time_out(reps);

  for (size_t i = 0; i < iterations; i++) {
    // reset thread sharing variables and output stream
    std::fill_n(out_stream.begin(), total_count, 0);

    // do an iteration
    high_resolution_clock::time_point start, end;
    DoOneIteration(q, buffers, buffer_count, reps, iterations, threads,
                   in_stream.data(), out_stream.data(),
                   time_in, time_out, start, end);

    // validate the results
    total_errors += CountErrors(out_stream.data(), total_count,
                                in_stream.data());

    // find the average latency for all reps
    double avg_rep_latency = 0.0;
    for (size_t j = 0; j < reps; j++) {
      duration<double, std::milli> l = time_out[j] - time_in[j];
      avg_rep_latency += l.count();
    }
    latency_ms[i] = avg_rep_latency / reps;

    // track the total processing time
    duration<double, std::milli> process_time = end - start;
    process_time_ms[i] = process_time.count();
  }

  // print the performance info
  PrintPerformanceInfo<T>("without API", total_count, latency_ms, process_time_ms);

  return total_errors == 0;
}

//
// The producer thread function. All production of data happens in this function
// which is run in a separate CPU thread from the launching of kernels and 
// consumption of data.
//
template<typename T>
void ProducerThread(T* in_stream, size_t buffer_count, size_t reps, int threads,
                    std::atomic<bool>& data_valid, T*& out_ptr) {
  // In our case, the Producer's job is simple. It has an input stream of data
  // ('in_stream') whose size is buffer_count * reps elements
  // (i.e. ALL of the data).
  // When signalled to, it produces buffer_count elements to 'out_ptr', which is
  // a shared variable between the Producer thread and the thread launching the
  // kernels.
  size_t rep = 0;
  while (rep < reps) {
    // The 'data_valid' flag is also shared between the Producer thread
    // and the kernel launching thread. The kernel thread sets 'data_valid' to 
    // false when it is ready for the Producer to produce new data to 'out_ptr'.
    // The 'data_valid' flag is the mechanism by which the Producer thread
    // tells the kernel launching thread that the input data is ready, AND how
    // the kernel launcher thread tells the Producer thread that it is ready
    // for new data.
    if (!data_valid) {
      // The kernel has signalled to the Producer that it is ready for new data,
      // so copy 'buffer_count' elements to the shared pointer 'out_ptr'
      ProducerFunction(out_ptr,
                      in_stream + buffer_count*rep,
                      buffer_count,
                      threads);

      // once the data has been produced, raise the 'data_valid' flag, which
      // signals to the kernel launching thread that the data at 'out_ptr' is
      // valid and ready to use. Remember, the kernel launching thread will
      // lower this flag when it is ready for new data to be produced!
      data_valid = true;

      rep++;
    }
  }
}

//
// This function handles both the launching of SYCL kernels and the
// consumption of the data.
//
template<typename T>
void KernelThread(queue& q, size_t buffers, size_t buffer_count, size_t reps,
                  int threads,
                  std::vector<T*>& in_buf, std::vector<T*>& out_buf,
                  std::atomic<bool>& produce_data_valid, T*& in_ptr,
                  T* out_stream,
                  std::vector<high_resolution_clock::time_point>& time_in,
                  std::vector<high_resolution_clock::time_point>& time_out) {
  // initialize
  size_t in_rep = 0;
  size_t out_rep = 0;
  size_t buf_idx = 0;

  // queue to track the inflight kernel events and the index of the buffer
  // they are using (for multi-buffering).
  std::queue<std::pair<event, size_t>> user_kernel_q;

  // do work
  while(out_rep < reps) {
    // Conditions for processing new input (launching a new kernel):
    // (NOTE: conditions 1, 2, AND 3 must be met)
    //  1) there is room in the queue (based on number of buffers)
    //  2) we have more input data to process
    //  3) the input data from the producer is valid (set by Producer thread)
    //
    // Conditions for consuming kernel output (waiting on oldest kernel to end):
    // (NOTE: conditions 1 OR 2 need to be met)
    //  1) the queue is full
    //  2) we have processed all of the input data (no kernels left to launch)
    bool queue_full = (user_kernel_q.size() == buffers);
    bool all_input_data_processed = (in_rep == reps);
    
    if (!queue_full && !all_input_data_processed && produce_data_valid) {
      // launch the kernel
      event e = SubmitKernel(q, in_buf[buf_idx], buffer_count, out_buf[buf_idx]);

      // push the new kernel event and buffer index pair into the queue
      user_kernel_q.push(std::make_pair(e, buf_idx));

      // mark the input time of this buffer (to track the latency)
      time_in[in_rep] = high_resolution_clock::now();

      // move to the next buffer (n-way buffering)
      buf_idx = (buf_idx + 1) % buffers;

      // if after pushing the new kernel there is still space in the queue,
      // then tell the producer we are ready to accept new data in the
      // next buffer
      if (user_kernel_q.size() < buffers) {
        // NOTE: order important for these two statements (described later)
        in_ptr = in_buf[buf_idx];
        produce_data_valid = false;
      }

      // we started the processing of another input
      in_rep++;
    } else if (queue_full || all_input_data_processed) {
      // pop the oldest event/buffer index pair in the queue
      auto event_index_pair = user_kernel_q.front();
      user_kernel_q.pop();

      // wait on the kernel event to finish
      event_index_pair.first.wait();

      // mark the output time of this buffer (to track the latency)
      time_out[out_rep] = high_resolution_clock::now();

      // tell the Producer that it can start producing the next input
      // data BEFORE consuming the output, so that the Producer can start
      // producing the next data while this thread consumes the output
      // (which we are about to do below).
      // NOTE: the order is important here. Switch the 'in_ptr' to the other
      // buffer BEFORE lowering the 'produce_data_valid' flag (which signals to
      // the Producer to start producing data into 'in_ptr'). I.e. we need to
      // set the correct pointer BEFORE signalling to the Producer to produce
      // to that pointer.
      in_ptr = in_buf[buf_idx];
      produce_data_valid = false;

      // consume the output
      // (NOTE: the kernel launching and Consumer use the same thread)
      ConsumerFunction(out_stream + out_rep*buffer_count,
                       out_buf[event_index_pair.second],
                       buffer_count,
                       threads);

      // we have processed another output
      out_rep++;
    }
  }
}

// Forward declare the kernel name to reduce name mangling
class Kernel;

//
// Submit the kernel for the single-kernel design
//
template<typename T>
event SubmitKernel(queue &q, T *in_ptr, size_t count, T *out_ptr) {
  auto e = q.submit([&](handler& h) {
    h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
      // using a host_ptr class tells the compiler that this pointer lives in
      // the host's address space
      host_ptr<T> in(in_ptr);
      host_ptr<T> out(out_ptr);

      for (size_t i = 0; i < count; i++) {
        T data = *(in + i);
        *(out + i) = data;
      }
    });
  });

  return e;
}

#endif /* __STREAMING_WITHOUT_API_HPP__ */