#ifndef __HOSTSTREAMER_HPP__
#define __HOSTSTREAMER_HPP__

#include <assert.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <functional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// A thread safe wrapper around std::queue.
// FUTURE WORK: We  could probably use conditional variables to improve
//              lock performance.
//
template<typename T>
class ConcurrentQueue {
private:
  std::queue<T> q_;
  std::mutex mtx_;

public:
  bool Empty() {
    std::scoped_lock lock(mtx_);
    return q_.size() == 0;
  }

  size_t Size() {
    std::scoped_lock lock(mtx_);
    return q_.size();
  }

  void Pop() {
    std::scoped_lock lock(mtx_);
    q_.pop();
  }

  T& Front() {
    std::scoped_lock lock(mtx_);
    return q_.front();
  }

  void Push(const T &data) {
    std::scoped_lock lock(mtx_);
    q_.push(data);
  }

  // these are useful in case the user of the queue wants to lock/unlock
  // the entire queue themselves
  std::mutex& GetMutex() { return mtx_; }
  void Lock() { mtx_.lock(); }
  void Unlock() { mtx_.unlock(); }
};

// Declare these out of the HostStreamer to reduce name mangling
template<typename Id>
class ProducerKernelId;
template<typename Id>
class ProducerPipeId;
template<typename Id>
class ConsumerKernelId;
template<typename Id>
class ConsumerPipeId;

//
// This class is used to stream to and/or from the host and device.
// It provides a set of APIs for high throughput, and a set of APIs
// for low latency.
//
// Template parameters:
//    Id:                     The unique ID for the HostStreamer. This is
//                            necessary since the pipes used to stream data
//                            to/from the device must be unique. Having multiple
//                            instances of this class would make sense.
//    ProducerType:           The datatype to stream from the host to the device
//    ConsumerType:           The datatype to stream from the device to the host
//    min_producer_capacity:  The minimum capacity of the ProducerPipe
//    min_consumer_capacity:  The minimum capacity of the ConsumerPipe
//
//  Using the HostStreamer results in a CPU-FPGA system that looks like this:
//
//  |------------|   |---------------------------------------------|
//  | <CPU>      |   | <FPGA>                     |--------------| |
//  |   |-----|  |   |  |----------| ProducerPipe |              | |
//  |   |     |--|---|->| Producer |==============|=> ...        | |
//  |   |     |  |   |  |----------|              |              | |
//  |   | USM |  |   |                            | User Kernels | |
//  |   |     |  |   |  |----------| ConsumerPipe |              | |
//  |   |     |<-|---|--| Consumer |<=============|== ...        | |
//  |   |-----|  |   |  |----------|              |              | |
//  |            |   |                            |--------------| |
//  |------------|   |---------------------------------------------|
//
// The interaction with USM and the Producer and Consumer kernels are abstracted
// from the user's device code ('User Kernels'). The abstraction is that input
// data (type=ProduceType) will be streamed from the host to the device through
// the ProducerPipe (HostStreamer<...>::ProducerPipe) and/or streamed from the
// device to the host through the ConsumerPipe (HostStreamer<...>::ProducerPipe)
// 
template <typename Id, typename ProducerType, typename ConsumerType,
          size_t min_producer_capacity=0, size_t min_consumer_capacity=0>
class HostStreamer {
private:
  // The constructor is private to avoid creating an instance of the class
  // The point of this is that every HostStreamer with a given 'Id' (the first
  // template parameter) is associated with a single Producer and Consumer pipe,
  // which are static. Therefore, having multiple instances of the same
  // HostStreamer doesn't make sense. To have multiple streaming inputs/outputs,
  // use multiple instances of HostStreamer with different 'Id' template
  // parameters, like you would when using SYCL pipes.
  HostStreamer() {}

  // Producer specific data structures
  static inline std::vector<ProducerType*> producer_buffer_{};
  static inline size_t num_producer_buffers_{};
  static inline size_t producer_buffer_size_{};
  static inline std::map<ProducerType*, size_t> producer_ptr_to_idx_map_{};
  static inline size_t producer_buffer_idx_{};

  // Consumer specific data structures
  static inline std::vector<ConsumerType*> consumer_buffer_{};
  static inline size_t num_consumer_buffers_{};
  static inline size_t consumer_buffer_size_{};
  static inline size_t consumer_buffer_idx_{};

  // These counters track the number of outstanding produce and consume
  // requests, respectively. Requests are outstanding from the time the Producer
  // acquires the pointer or the Consumer launches the read, until the
  // the KernelLaunchAndWaitThread waits on the kernel event associated with
  // the request. Each counter has an associated mutex for thread safety.
  static inline size_t produce_requests_outstanding_{};
  static inline std::mutex produce_requests_outstanding_mtx_{};
  static inline size_t consume_requests_outstanding_{};
  static inline std::mutex consume_requests_outstanding_mtx_{};

  // The Producer and Consumer queues. Produce and Consume events
  // from user API calls first go into these queues, respectively.
  //
  // producer_consumer_tuple = 
  //      <size_t: index into producer_buffer or consumer buffer,
  //       size_t: the count of elements to be produced/consumer>
  using producer_consumer_tuple = std::tuple<size_t, size_t>;
  static inline ConcurrentQueue<producer_consumer_tuple> produce_q_{};
  static inline ConcurrentQueue<producer_consumer_tuple> consume_q_{};

  // The KernelLaunchAndWaitThread grabs requests from the Producer and Consumer
  // queues (declared above) and places them into the launch queue. From there
  // the KernelLaunchAndWaitThread grabs requests from the launch queue, and 
  // adds them to the actual SYCL queue by launching the necessary Producer or
  // Consumer SYCL kernel.
  //
  // launch_queue_tuple = 
  //      <size_t: index into producer_buffer or consumer buffer,
  //       size_t: the count of elements to be produced/consumer,
  //       event: the SYCL event for the launched kernel
  //       bool: true for producer, false for consumer>
  using launch_queue_tuple = std::tuple<size_t, size_t, event, bool>;
  static inline ConcurrentQueue<launch_queue_tuple> launch_q_{};

  // A pointer to the SYCL queue which launches the actual kernels to do the
  // producing and consuming. We don't use a reference here due to static
  // initialization issues.
  static inline queue* sycl_q_{};

  // This is the number of kernels (both Producer and Consumer kernels)
  // that we want to have in-flight (i.e. in the SYCL queue) before waiting on
  // the oldest event to finish. Setting this too low (e.g. 1) will result in
  // us NOT taking advantage of fast kernel relaunch and a drop in throughput.
  // Setting this too high (e.g. 2000) will result in kernels that are in-flight
  // finishing execution before we call wait (event.wait()) on them.
  // This too will result in a drop in throughput.
  static inline size_t wait_threshold_{};

  // Signals to the KernelLaunchAndWaitThread to flush the launch queue
  static inline std::atomic<bool> flush_{false};

  // This breaks the while() loop in the KernelLaunchAndWaitThread.
  // this allows the KernelLaunchAndWaitThread to be safely terminated so the
  // main thread can join with it.
  static inline std::atomic<bool> kill_kernel_thread_flag_{false};

  // A pointer to the KernelLaunchAndWaitThread C++ thread object
  static inline std::thread *kernel_thread_{nullptr};

  // track whether the single instance has been initialized or not
  static inline bool initialized_{false};

  // Convenience methods for querying the status of the Producer, Consumer,
  // and Launch queues
  static bool ProducerQueueFull() {
    return produce_q_.Size() == num_producer_buffers_;
  }
  static bool ProducerQueueEmpty() {
    return produce_q_.Empty();
  }
  static bool ConsumerQueueFull() {
    return consume_q_.Size() == num_consumer_buffers_;
  }
  static bool ConsumerQueueEmpty() {
    return consume_q_.Empty();
  }
  static bool LaunchQueueEmpty() {
    return launch_q_.Empty();
  }

  // This function will run in a separate CPU thread. It's job is to grab
  // produce and consume requests from the Producer and Consumer queue
  // (produce_q_ and consume_q_, respectively), merge them into a single request
  // queue (launch_q_), and finally launch the actual SYCL kernels into the SYCL
  // queue (sycl_q_) to perform the request. It also performs the callbacks
  // to the user code when the requests have been completed.
  static void KernelLaunchAndWaitThread() {
    // Do this loop until told (by main thread) to stop via the
    // 'kill_kernel_thread_flag_' atomic shared variable.
    while (!kill_kernel_thread_flag_) {
      // If there is a Produce request to launch, do it
      if (!ProducerQueueEmpty()) {
        // grab the oldest request from the produce queue
        size_t buf_idx;
        size_t count;
        std::tie(buf_idx, count) = produce_q_.Front();

        // launch the kernel and push the request to the launch queue
        auto e = LaunchProducerKernel(producer_buffer_[buf_idx], count);
        launch_q_.Push(std::make_tuple(buf_idx, count, e, true));

        // pop from the Producer queue
        produce_q_.Pop();
      }

      // If there is a Consume request to launch, do it
      if (!ConsumerQueueEmpty()) {
        // grab the oldest request from the consume queue
        size_t buf_idx;
        size_t count;
        std::tie(buf_idx, count) = consume_q_.Front();

        // launch the kernel and push the request to the launch queue
        auto e = LaunchConsumerKernel(consumer_buffer_[buf_idx], count);
        launch_q_.Push(std::make_tuple(buf_idx, count, e, false));

        // pop from the Consumer queue
        consume_q_.Pop();
      }

      // Wait on the oldest event to finish given 2 conditions:
      //    1) there are a certain number of kernels in-flight
      //       (i.e. launch_q_.size() >= wait_threshold_)
      //    2) the user has requested us to flush the launch queue and the
      //       launch queue is not empty (i.e. flush_ && launch_q_.size() != 0)
      if ((launch_q_.Size() >= wait_threshold_) ||
          (flush_ && !LaunchQueueEmpty())) {
        // grab the oldest request from the launch queue
        size_t buf_idx;
        size_t count;
        event e;
        bool request_was_producer;
        std::tie(buf_idx, count, e, request_was_producer) = launch_q_.Front();

        // wait on the oldest event to finish
        e.wait();

        // call the appropriate callback
        if (request_was_producer) {
          //std::cout << "Calling Producer Callback" << std::endl;
          producer_callback(count);
        } else {
          //std::cout << "Calling Consumer Callback" << std::endl;
          consumer_callback(consumer_buffer_[buf_idx], count);
        }

        // Pop from the launch queue. This has to be done AFTER waiting on
        // the SYCL kernel event and calling the callback.
        launch_q_.Pop();

        // We just acted upon a request by launching the kernel
        // (at some earlier time), waiting on the kernel, and acting on the 
        // data via a callback. Therefore, the request is complete! So reduce
        // the number of outstanding requests for the Producer or Consumer
        // appropriately. Don't forget the (correct) lock!
        if (request_was_producer) {
          ////////////////////////////////////////
          // Entering critical section
          produce_requests_outstanding_mtx_.lock();

          assert(produce_requests_outstanding_ > 0);
          produce_requests_outstanding_--;
          
          produce_requests_outstanding_mtx_.unlock();
          // Exiting critical section
          ////////////////////////////////////////
        } else {
          ////////////////////////////////////////
          // Entering critical section
          consume_requests_outstanding_mtx_.lock();

          assert(consume_requests_outstanding_ > 0);
          consume_requests_outstanding_--;
          
          consume_requests_outstanding_mtx_.unlock();
          // Exiting critical section
          ////////////////////////////////////////
        }
      }
    }
  }

public:
  // The Producer and Consumer SYCL pipes.
  // This allows device code (i.e. user kernels) to connect to the input and
  // the output.
  using ProducerPipe = sycl::INTEL::pipe<ProducerPipeId<Id>,
                                         ProducerType,
                                         min_producer_capacity>;
  using ConsumerPipe = sycl::INTEL::pipe<ConsumerPipeId<Id>,
                                         ConsumerType,
                                         min_consumer_capacity>;

  // The user can query the input and output types of the pipes
  // E.g.
  // using MyStreamer = HostStreamer<class MyStreamerClass, int, 32, float, 32>;
  // MyStreamer::produce_type (=int)
  // MyStreamer::consume_type (=float)
  using produce_type = ProducerType;
  using consume_type = ConsumerType;

  // The callback functions for Producer and Consumer, respectively.
  // By default, they are empty functions that do nothing. It is the user's job
  // to specify their own callback functions.
  // NOTE: the user will certainly want to capture the 'consumer_callback',
  // but may not care about capturing the 'producer_callback'.
  static inline std::function<void(size_t)>
    producer_callback = [](size_t) {};
  static inline std::function<void(const ConsumerType*, size_t)>
    consumer_callback = [](const ConsumerType*, size_t) {};

  // getter and setter to override the maximum number of kernels in-flight
  static inline size_t wait_threshold() { return wait_threshold_; }
  static inline void wait_threshold(size_t wt) { wait_threshold_ = wt; }

  //////////////////////////////////////////////////////////////////////////////
  // Initialization
  static void init(queue& q,
                   size_t num_producer_buffers=2,
                   size_t producer_buffer_size=65536,
                   size_t num_consumer_buffers=2,
                   size_t consumer_buffer_size=65536) {
    // if already initialized, deal with teardown first
    // NOTE: must do this before re-initialzing
    if (initialized_) {
      std::cout << "WARNING: HostStreamer<...>::init() was called without "
                << "HostStreamer<...>::destroy() being called first. "
                << "Reinitializing.\n";

      // For now, simply destroy first.
      // FUTURE WORK: we could probably do something smart by looking
      // at the change in the device queue, the number of producer/consumer
      // buffers, and the size of those buffers. But this will likely not
      // affect performance much, so we will just destroy for now.
      destroy();
    }

    // save a pointer to the SYCL queue
    sycl_q_ = &q;

    // the number of Producer buffers is the number of Producer kernels that
    // we can have in-flight at once (since the operate on different buffers).
    // Likewise for the Consumer buffers. Therefore, the number of kernels
    // that we want in-flight at once (determined by the wait_threshold_)
    // is determined by the total number of Producer and Consumer buffers.
    wait_threshold_ = num_producer_buffers + num_consumer_buffers;

    //////////////////////////////////////////////
    // Producer
    num_producer_buffers_ = num_producer_buffers;
    producer_buffer_size_ = producer_buffer_size;
    producer_buffer_.resize(num_producer_buffers_);
    producer_buffer_idx_ = 0;

    // allocate USM space for buffers
    for (auto& b : producer_buffer_) {
      b = malloc_host<ProducerType>(producer_buffer_size_, *sycl_q_);
      if (b == nullptr) {
        std::cerr << "Could not allocate USM memory for producer buffer\n";
        std::terminate();
      }
    }

    // build the USM pointer->buffer index map
    for (size_t i = 0; i < num_producer_buffers_; i++) {
      producer_ptr_to_idx_map_[producer_buffer_[i]] = i;
    }

    produce_requests_outstanding_ = 0;
    //////////////////////////////////////////////

    //////////////////////////////////////////////
    // Consumer
    num_consumer_buffers_ = num_consumer_buffers;
    consumer_buffer_size_ = consumer_buffer_size;
    consumer_buffer_.resize(num_consumer_buffers_);
    consumer_buffer_idx_ = 0;

    // allocate USM space for buffers
    for (auto& b : consumer_buffer_) {
      b = malloc_host<ConsumerType>(consumer_buffer_size_, *sycl_q_);
      if (b == nullptr) {
        std::cerr << "Could not allocate USM memory for consumer buffer\n";
        std::terminate();
      }
    }

    consume_requests_outstanding_ = 0;
    //////////////////////////////////////////////

    // start the KernelLaunchAndWaitThread
    flush_ = false;
    kill_kernel_thread_flag_ = false;
    kernel_thread_ = new std::thread(&KernelLaunchAndWaitThread);

    // have been initialized
    initialized_ = true;
  }

  // Destruction
  // must be the last API call made
  static void destroy() {
    // make sure we have initialized the HostStreamer
    if (initialized_) {
      // stop the kernel thread safely, join with it, and destroy the thread
      kill_kernel_thread_flag_ = true;
      kernel_thread_->join();
      delete kernel_thread_;
      kernel_thread_ = nullptr;

      // free all the USM memory
      for (auto& b : producer_buffer_) {
        sycl::free(b, *sycl_q_);
        b = nullptr;
      }
      for (auto& b : consumer_buffer_) {
        sycl::free(b, *sycl_q_);
        b = nullptr;
      }

      // clear the buffer pointer -> idx map
      producer_ptr_to_idx_map_.clear();

      // nullptr the SYCL queue pointer
      sycl_q_ = nullptr;

      // no longer initialized
      initialized_ = false;
    } else {
      std::cout << "WARNING: HostStreamer<...>::destroy() called on an "
                << "uninitialized HostStreamer. Nothing to do.\n";
    }
  }
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // high throughput, high(er) latency API

  // The user calls this function to attempt at acquiring a buffer.
  // If a buffer is available for use, this function returns it. Otherwise
  // it returns nullptr. The point of the acquire/release for producing data
  // is to avoid copying data from the user into USM buffers. Giving them the
  // pointers allows them to produce the data directly into the USM buffers.
  //
  // FUTURE WORK: Could probably improve the locking here
  static ProducerType* AcquireProducerBuffer() {
    ProducerType *acquired_ptr = nullptr;

    ///////////////////////////////////
    // Entering critical section
    // this allows AcquireProducerBuffer to be called from different threads
    produce_requests_outstanding_mtx_.lock();

    // if we have room for another produce request
    if (produce_requests_outstanding_ < num_producer_buffers_) {
      // There is room for another produce request grab the 'head' produce
      // buffer, move to the next buffer, and increment the number of produce
      // requests outstanding
      acquired_ptr = producer_buffer_[producer_buffer_idx_];
      producer_buffer_idx_ = (producer_buffer_idx_ + 1) % num_producer_buffers_;
      produce_requests_outstanding_++;
    }

    produce_requests_outstanding_mtx_.unlock();
    // Exiting critical section
    ///////////////////////////////////

    return acquired_ptr;
  }

  // The user calls this function to release a previously acquired buffer
  // (via a call to AcquireProducerBuffer) back to the API. This implies that
  // the user has produced their data into the buffer and are ready for 
  // the request to continue (i.e. the kernel to produce the data to be
  // launched). The user may not want to use all of the buffer (all 
  // 'producer_buffer_size_; elements). The 'release_size' argument allows
  // them to specificy how much data they produced into the buffer, which should
  // be less than or equal to the size of the buffer ('producer_buffer_size_').
  static void ReleaseProducerBuffer(ProducerType* acquired_ptr,
                                    size_t release_size) {
    // error checking
    if (ProducerQueueFull()) {
      std::cerr << "ERROR: ReleaseProducerBuffer was called but "
                << "the Producer queue is full. This should not be possible. "
                << "This could be caused by calling ReleaseProducerBuffer more " 
                << "than once for the same pointer returned by "
                << "AcquireProducerBuffer\n";
      std::terminate();
    }

    // error checking
    if (release_size > producer_buffer_size_) {
      std::cerr << "ERROR: tried to write " << release_size << " elements but "
                << "the buffer size is only " << producer_buffer_size_ << "\n";
      std::terminate();
    }
    
    // find the buffer index based on the pointer
    auto it = producer_ptr_to_idx_map_.find(acquired_ptr);

    // error checking, make sure the pointer to release is actually one of the
    // buffers that were acquired.
    if (it == producer_ptr_to_idx_map_.end()) {
      std::cerr << "ERROR: an unknown pointer was passed to "
                << "ReleaseProducerBuffer.\n";
      std::terminate();
    }

    // get the buffer index from the iterator
    size_t buf_idx = it->second;

    // push the produce request
    produce_q_.Push(std::make_tuple(buf_idx, release_size));
  }

  // This single API call is used by the user to create a consume request.
  // The return boolean indicates whether the request was accepted or not (based
  // on the number of outstanding consume requests). If the call succeeds (i.e.
  // this function returns 'true') then the reques was accepted and the 
  // 'consumer_callback' function will be called sometime in the future by the
  // API as a response to this request.
  static bool RequestConsumer(size_t launch_size) {
    bool success;

    ///////////////////////////////////
    // Entering critical section
    // this allows AcquireProducerBuffer to be called from different threads
    consume_requests_outstanding_mtx_.lock();

    if (consume_requests_outstanding_ >= num_consumer_buffers_) {
      // full of reading consume events, failed to do a new one
      assert(consume_requests_outstanding_ == num_consumer_buffers_);
      success = false;
    } else {
      // error checking
      if (launch_size > consumer_buffer_size_) {
        std::cerr << "ERROR: tried to read " << launch_size << " elements but "
                  << "the buffer size is only " << consumer_buffer_size_
                  << "\n";
        std::terminate();
      }

      // error checking
      if (ConsumerQueueFull()) {
        std::cerr << "ERROR: LaunchConsumer was called and was about to launch "
                  << "a new Consumer kernel, but the Consumer queue is full\n";
        std::terminate();
      }

      // Push the consume request, move to the next Consumer buffer, increment
      // the number of outstanding Consume requests, and set the success code.
      consume_q_.Push(std::make_tuple(consumer_buffer_idx_, launch_size));
      consumer_buffer_idx_ = (consumer_buffer_idx_ + 1) % num_consumer_buffers_;
      consume_requests_outstanding_++;
      success = true;
    }

    consume_requests_outstanding_mtx_.unlock();
    // Exiting critical section
    ///////////////////////////////////

    return success;
  }

  // Tell the KernelLaunchAndWaitThread to flush the launch queue.
  static void Flush() {
    flush_ = true;
  }

  // This synchronizes with the KernelLaunchAndWaitThread. This should be called
  // This should be called once the user is done performing ALL request
  // (i.e. both Produce and Consume request).
  // NOTE: this API call is blocking. It will block until all of the kernels
  // have been launched and finished.
  static void Sync() {
    // flush the launch queue
    Flush();

    // wait until the launch queue is empty
    while (!LaunchQueueEmpty()){}
  }
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // low latency, low throughput API
  // these functions behave a lot like host pipes and are convenient to use
  // if one does not care about throughput
  static void Write(const ProducerType &data) {
    // write the data into the first buffer
    producer_buffer_[0][0] = data;

    // launch kernel to produce a single element of data to the ProducerPipe
    auto e = LaunchProducerKernel(producer_buffer_[0], 1);

    // wait for kernel to finish before returning to user (synchronous)
    e.wait();
  }

  static ConsumerType Read() {
    // launch a kernel to read one element from the ConsumerPipe
    auto e = LaunchConsumerKernel(consumer_buffer_[0], 1);

    // wait for kernel to finish before returning to user (synchronous)
    e.wait();

    // return the data read
    return consumer_buffer_[0][0];
  }
  //////////////////////////////////////////////////////////////////////////////
  
  //////////////////////////////////////////////////////////////////////////////
  // Kernel functions
  // NOTE: the code in these functions are device code. This means they get
  // synthesized into FPGA kernels.
  static event LaunchProducerKernel(ProducerType *usm_ptr, size_t count) {
    return sycl_q_->submit([&](handler& h) {
      h.single_task<ProducerKernelId<Id>>([=]() {
        host_ptr<ProducerType> ptr(usm_ptr);
        for (size_t i = 0; i < count; i++) {
          // host->device: read from USM and write to the ProducerPipe
          auto val = *(ptr + i);
          ProducerPipe::write(val);
        }
      });
    });
  }

  static event LaunchConsumerKernel(ConsumerType *usm_ptr, size_t count) {
    return sycl_q_->submit([&](handler& h) {
      h.single_task<ConsumerKernelId<Id>>([=]() {
        host_ptr<ConsumerType> ptr(usm_ptr);
        for (size_t i = 0; i < count; i++) {
          // device->host: read from the ConsumerPipe and write to USM
          auto val = ConsumerPipe::read();
          *(ptr + i) = val;
        }
      });
    });
  }
  //////////////////////////////////////////////////////////////////////////////
};

#endif /* __HOSTSTREAMER_HPP__ */