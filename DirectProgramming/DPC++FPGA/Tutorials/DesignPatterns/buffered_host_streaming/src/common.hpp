#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>
#include <thread>

// A multithreaded version of memcpy.
// On modern processors with multiple cores and threads, a single-threaded
// memcpy cannot saturate the memory bandwidth (not even close). Moreover,
// a single-threaded memcpy doesn't even get close to the PCIe bandwidth.
// To improve the performance of the Producer and Consumer functions
// (which are simply memcpy in our simple design) we use a multi-threaded
// version of memcpy.
template<typename T>
void memcpy_threaded(T* dst, const T* src, size_t count, int num_threads) {
  if (num_threads == 1) {
    // if the number of threads is 1, just do a regular memcpy()
    memcpy(dst, src, count * sizeof(T));
  } else {
    // multi-threaded memcpy()
    std::vector<std::thread> threads;

    // number of elements per thread = ceil(count/num_threads)
    size_t count_per_thread = (count + num_threads - 1) / num_threads;

    // the last thread may have a different count if 'num_threads' is not 
    // a multiple of 'count'.
    size_t count_last_thread = count - (num_threads-1)*count_per_thread;

    // thread lambda function
    auto f = [](T* dst, const T* src, size_t count) {
      memcpy(dst, src, count * sizeof(T));
    };

    // launch the threads
    for (int i = 0; i < num_threads; i++) {
      size_t t_count = (i == num_threads-1) ? count_last_thread
                                            : count_per_thread;
      threads.push_back(std::thread(f,
                                    dst + i*count_per_thread,
                                    src + i*count_per_thread,
                                    t_count));
    }

    // wait for the threads to finish
    for (auto& t : threads) {
      t.join();
    }
  }
}

// The Producer function is a simple memcpy() from input to output
template<typename T>
void ProducerFunction(T* dst, const T* src, size_t count, int num_threads) {
  memcpy_threaded(dst, src, count, num_threads);
}

// The Consumer function is a simple memcpy() from input to output
template<typename T>
void ConsumerFunction(T* dst, const T* src, size_t count, int num_threads) {
  memcpy_threaded(dst, src, count, num_threads);
}

// A helper function to compute and print the performance info
template<typename T>
void PrintPerformanceInfo(std::string postfix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms) {
  // compute the input size in MB
  double input_size_mb = (sizeof(T) * count) * 1e-6;
  // compute the average latency and processing time
  assert(latency_ms.size() == process_time_ms.size());
  double iterations = latency_ms.size() - 1;
  double avg_latency_ms = std::accumulate(latency_ms.begin() + 1,
                                          latency_ms.end(),
                                          0.0) / iterations;
  double avg_processing_time_ms = std::accumulate(process_time_ms.begin() + 1,
                                                  process_time_ms.end(),
                                                  0.0) / iterations;
  // compute the throughput
  double avg_tp_mb_s = input_size_mb / (avg_processing_time_ms * 1e-3);
  // print info
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Average latency " << postfix << ": "
            << avg_latency_ms << " ms\n";
  std::cout << "Average processing time " << postfix << ": " 
            << avg_processing_time_ms << " ms\n";
  std::cout << "Average throughput " << postfix << ": "
            << avg_tp_mb_s  << " MB/s\n";
}

// given an output ('out') and a reference ('ref') count the number of errors
template<typename T>
size_t CountErrors(T* out, size_t count, T* ref) {
  size_t errors = 0;
  for (size_t i = 0; i < count; i++) {
    auto comp = (out[i] == ref[i]);
    for (auto j = 0; j < comp.get_count(); j++) {
      if (!comp[j]) {
        errors++;
      }
    }
  }
  return errors;
}

#endif /* __COMMON_HPP__ */
