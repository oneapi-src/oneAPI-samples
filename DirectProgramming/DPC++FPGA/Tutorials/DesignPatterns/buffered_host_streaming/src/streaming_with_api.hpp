#ifndef __STREAMING_WITH_API_HPP__
#define __STREAMING_WITH_API_HPP__

#include <algorithm>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "common.hpp"
#include "HostStreamer.hpp"

using namespace sycl;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
// forward declare functions
template<typename T>
void DoOneIterationAPI(queue& q, size_t buffers, size_t buffer_count,
                       size_t reps, size_t iterations, size_t threads,
                       T *in_stream, T *out_stream,
                       std::vector<high_resolution_clock::time_point>& time_in,
                       std::vector<high_resolution_clock::time_point>& time_out,
                       high_resolution_clock::time_point& start,
                       high_resolution_clock::time_point& end);

template<typename T>
bool DoWorkAPI(queue& q, size_t buffers, size_t buffer_count, size_t reps,
               size_t iterations, size_t threads);
////////////////////////////////////////////////////////////////////////////////

// Forward declare the kernel and HostStreamer name to reduce name mangling
class APIKernel;
class MyStreamerId;

//
// This function contains the logic for buffered streaming.
// It perform a single iteration of the design. The calling function will
// call this function multiple times to increase the performance
// measurement accuracy.
// 
template<typename T>
void DoOneIterationAPI(queue& q, size_t buffers, size_t buffer_count,
                       size_t reps, size_t iterations, size_t threads,
                       T *in_stream, T *out_stream,
                       std::vector<high_resolution_clock::time_point>& time_in,
                       std::vector<high_resolution_clock::time_point>& time_out,
                       high_resolution_clock::time_point& start,
                       high_resolution_clock::time_point& end) {
  // The Producer and Consumer get half of the total threads, each.
  // std::max() guards against the case where there is only 1 thread.
  size_t half_threads = std::max(size_t(1), threads/2);

  // total number of elements
  size_t total_count = buffer_count * reps;

  // Alias 'MyStreamer' to the templated HostStreamer. The streamer
  // will stream in and out elements of type 'T'.
  //    Template arguments for HostStreamer (in order)
  //        Id:                     MyStreamerId
  //        ProducerType:           T
  //        ConsumerType:           T
  //        min_producer_capacity:  0 (implicit)
  //        min_consumer_capacity:  0 (implicit)
  using MyStreamer = HostStreamer<MyStreamerId, T, T>;

  // Initialize the streamer
  //    # of Producer buffers      = 'buffers'
  //    size of Producer buffers   = 'buffer_count'
  //    # of Consumer buffers      = 'buffers'
  //    size of Consumer buffers   = 'buffer_count'
  MyStreamer::init(q, buffers, buffer_count, buffers, buffer_count);

  //////////////////////////////////////////////////////////////////////////
  // Setup the Producer and Consumer callback function for the HostStreamer.

  // This consumer_callback function is called when the kernel, which is
  // launched by a consumer request (i.e. MyStreamer::RequestConsumer),
  // completes and the output data (ptr) is ready to be processed by the host.
  // There will be one call to this callback for every successful call to 
  // MyStreamer::RequestConsumer.
  int out_rep = 0;
  MyStreamer::consumer_callback = [&](const T* ptr, size_t count) {
    // mark the time we received this data (for latency measurements)
    time_out[out_rep] = high_resolution_clock::now();

    // Consume the output
    // In this simple design, the Consumer simply copies the output (ptr)
    // into a larger buffer (out_stream). In a 'real' system, this may
    // be moving the data somewhere else, or performing more computation on it.
    ConsumerFunction(&out_stream[out_rep*buffer_count],
                     ptr,
                     count,
                     half_threads);

    // next repetition
    out_rep++;
  };

  // In our case, we don't really care about the producer_callback. It is
  // called when the kernel that did the producing finishes. The API defaults
  // the callback to an empty function, so we could omit the code that sets
  // 'producer_callback' below, but we include it for completeness.
  MyStreamer::producer_callback = [&](size_t /*count*/) {
    // nothing to do...
  };
  //////////////////////////////////////////////////////////////////////////

  // Launch the FPGA processing kernel. This kernel is only launched ONCE!
  // Therefore, it has to be able to process ALL of the data. In this example,
  // we know the total amount of data to be processed ('total_count') and
  // therefore we can easily bound the computation of this kernel. In other
  // cases, this may not be possible and an infinite loop may be required (i.e.
  // read from the Producer pipe and produce to the Consumer pipe, forever).
  auto kernel_event = q.submit([&](handler& h) {
    h.single_task<APIKernel>([=] {
      // process ALL of the possible data
      for (size_t i = 0; i < total_count; i++) {
        // read from the producer pipe
        auto data = MyStreamer::ProducerPipe::read();

        // <<<<< your computation goes here! >>>>>

        // write to the consumer pipe
        MyStreamer::ConsumerPipe::write(data);
      }
    });
  });
  
  start = high_resolution_clock::now();

  // Start the producer thread.
  // The code in the lamda runs in a different thread and therefore does not
  // block the process of the 'main' thread.
  std::thread producer_thread([&] {
    size_t rep = 0;
    T *buffer = nullptr;

    while (rep < reps) {
      // try to acquire the buffer from the HostStreamer API
      buffer = MyStreamer::AcquireProducerBuffer();

      // check if we acquired the buffer
      if (buffer != nullptr) {
        // If we acquired the buffer, produce to it.
        // In our design, we simply produce data by copying from a portion
        // of the larger 'in_stream' buffer to the buffer we just acquired
        // ('buffer'). In a 'real' design, this production may actual
        // computation (e.g. a random number generator), come from another
        // process, from an IO device (e.g. Ethernet), etc.
        ProducerFunction(buffer,
                         &in_stream[buffer_count*rep],
                         buffer_count,
                         half_threads);

        // The producing is done, release the buffer. This releases 'ownership'
        // of 'buffer' back to the API and creates a produce request to the API
        // to produce 'buffer_count' elements from 'buffer' to the device
        // (through MyStreamer::ProducerPipe).
        MyStreamer::ReleaseProducerBuffer(buffer, buffer_count);

        // mark input time for this rep (for latency measurements)
        time_in[rep] = high_resolution_clock::now();

        // rep done
        rep++;
      }
    }
  });

  // Make all of the asynchronous consume requests
  size_t in_rep = 0;
  while (in_rep < reps) {
    // This simply makes a Consumer request. If MyStreamer::RequestConsumer
    // returns 'false', the request was NOT accepted by the API and therefore 
    // you should try again. If MyStreamer::RequestConsumer returns 'true' the
    // request was accepted and, at some later time, you (the API user) will be
    // notified that the request is complete by the consumer_callback
    // (MyStreamer::consume_callback, defined earlier in this function)
    // which will provide the output data and size.
    if (MyStreamer::RequestConsumer(buffer_count)) {
      in_rep++;
    }
  }

  // Wait for the producer thread to finish. It will finish once all of the
  // produce requests have been launched (that doesn't mean they are done).
  producer_thread.join();

  // Synchronize with the HostStreamer. This will wait until the launch queue is
  // empty, which happens when all of the produce and consume requests have been
  // completed (i.e. kernel launched, complete, and callback complete).
  MyStreamer::Sync();

  // Wait on the main processing kernel event.
  // NOTE: if the kernel executed forever (i.e. a while(1)-loop) this would
  // block forever, and therefore should be omitted. The fact that all the
  // data was consumed by the Consumer (MyStreamer::Sync() returned)
  // implies the computation is done.
  kernel_event.wait();

  end = high_resolution_clock::now();

  // destroy the streamer data structures
  MyStreamer::destroy();
}

//
// The top level function for doing work with the API. It deals with timing,
// checking errors, and running multiple iterations to improve performance
// accuracy.
//
template<typename T>
bool DoWorkAPI(queue& q, size_t buffers, size_t buffer_count, size_t reps,
               size_t iterations, size_t threads) {
  // track how many errors we detect in the output
  size_t total_errors = 0;

  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // track the input and output times for each repetition (latency)
  std::vector<high_resolution_clock::time_point> time_in(reps);
  std::vector<high_resolution_clock::time_point> time_out(reps);

  // create input and output streams of data for ALL the data to be processed
  size_t total_count = buffer_count * reps;
  std::vector<T> in_stream(total_count);
  std::vector<T> out_stream(total_count);
  
  // generate random input data
  std::generate_n(in_stream.begin(), total_count, [] { return rand() % 100; });

  for (size_t i = 0; i < iterations; i++) {
    // reset stuff
    std::fill_n(out_stream.begin(), total_count, 0);

    // do the iteration
    high_resolution_clock::time_point start, end;
    DoOneIterationAPI(q, buffers, buffer_count, reps, iterations, threads,
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
  PrintPerformanceInfo<T>("with API", total_count, latency_ms, process_time_ms);

  return total_errors == 0;
}

#endif /* __STREAMING_WITH_API_HPP__ */
