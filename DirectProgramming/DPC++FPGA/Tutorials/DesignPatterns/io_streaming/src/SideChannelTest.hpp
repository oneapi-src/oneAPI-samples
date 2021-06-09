#ifndef __SIDECHANNELTEST_HPP__
#define __SIDECHANNELTEST_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "FakeIOPipes.hpp"
#include "HostSideChannel.hpp"

using namespace sycl;
using namespace std::chrono_literals;

// declare the kernel and pipe ID stucts globally to reduce name mangling
struct SideChannelMainKernel;
struct SideChannelIOPipes {
  struct ReadIOPipeID { static constexpr unsigned id = 0; };
  struct WriteIOPipeID { static constexpr unsigned id = 1; };
};
struct SideChannelPipes {
  struct HostToDeviceSideChannelID;
  struct DeviceToHostSideChannelID;
  struct HostToDeviceTermSideChannelID;
};

//
// Submit the main processing kernel (or kernels, in general).
// This function is templated on the IO pipes and the side channels.
// It is agnostic to whether the the system is using real or fake IO pipes.
//
// This kernel streams data into and out of IO pipes (IOPipeIn and IOPipeOut,
// respectively). If the value matches the current value of 'match_num', it
// sends the value to the host via the HostToDeviceSideChannel. In the outer
// loop, it reads configuration data from the host via the
// HostToDeviceSideChannel and HostToDeviceTermSideChannel. The former updates
// 'match_num' and the latter causes this kernel to break from the outer loop
// and terminate.
//
template<class IOPipeIn, class IOPipeOut,
         class HostToDeviceSideChannel, class DeviceToHostSideChannel,
         class HostToDeviceTermSideChannel>
event SubmitSideChannelKernels(queue& q, int initial_match_num,
                               size_t frame_size) {
  // the maximum number of consecutive input read misses before
  // breaking out of the computation loop
  // Why would you want to break out of the processing loop? One example
  // is if you are expecting asynchronous updates from the host through a
  // side channel.
  constexpr size_t kTimeoutCounterMax = 1024;

  // submit the main processing kernel
  return q.submit([&](handler& h) {
    h.single_task<SideChannelMainKernel>([=] {
      int match_num = initial_match_num;
      size_t timeout_counter;
      size_t samples_processed = 0;
      bool terminate = false;

      while (!terminate) {
        // check for an update to the sum_threshold from the host
        bool valid_update;
        int tmp = HostToDeviceSideChannel::read(valid_update);
        if (valid_update) match_num = tmp;

        // reset the timeout counter
        timeout_counter = 0;

        // if we processed a full frame, reset the counter
        // this places a maximum on the number of elements we process before
        // checking for an update from the host
        if (samples_processed == frame_size) samples_processed = 0;

        // do the main processing
        while ((samples_processed != frame_size) && 
               (timeout_counter != kTimeoutCounterMax)) {
          // read from the input IO pipe
          bool valid_read;
          auto val = IOPipeIn::read(valid_read);

          if (valid_read) {
            // reset the timeout counter since we read a valid piece of data
            timeout_counter = 0;

            // processed another sample in this frame
            samples_processed++;

            // check if the value matches
            if (val == match_num) {
              // value matches, so tell the host about it
              DeviceToHostSideChannel::write(val);
            }
            
            // propagate the input value to the output
            IOPipeOut::write(val);
          } else {
            // increment the timeout counter since the read was invalid
            timeout_counter++;
          }
        }

        // the host uses this side channel to tell the kernel to exit
        // Any successful read from this channel means the host wants this
        // kernel to end; the actual value read doesn't matter
        (void)HostToDeviceTermSideChannel::read(terminate);
      }
    });
  });
}

//
// This function builds the full system using fake IO pipes.
// It creates, produces, and consumes the fake data and validates the output
//
template<typename T, bool use_usm_host_alloc>
bool RunSideChannelsSystem(queue& q, size_t count) {
  //////////////////////////////////////////////////////////////////////////////
  // IO pipes
  // these are the FAKE IO pipes
  using FakeIOPipeInProducer = Producer<SideChannelIOPipes::ReadIOPipeID,
                                T, use_usm_host_alloc>;
  using FakeIOPipeOutConsumer = Consumer<SideChannelIOPipes::WriteIOPipeID,
                                 T, use_usm_host_alloc>;
  using ReadIOPipe = typename FakeIOPipeInProducer::Pipe;
  using WriteIOPipe = typename FakeIOPipeOutConsumer::Pipe;

  // initialize the fake IO pipes
  FakeIOPipeInProducer::Init(q, count);
  FakeIOPipeOutConsumer::Init(q, count);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // the side channels
  using MyHostToDeviceSideChannel = 
    HostToDeviceSideChannel<SideChannelPipes::HostToDeviceSideChannelID,
                            int, use_usm_host_alloc, 1>;
  using MyHostToDeviceTermSideChannel = 
    HostToDeviceSideChannel<SideChannelPipes::HostToDeviceTermSideChannelID,
                            char, use_usm_host_alloc, 1>;
  
  // This side channel is used to sent updates from the device to the host.
  // We explicitly set the depth of the FIFO to '8' here. If the host does not
  // read from this side channel quick enough it will causes the main processing
  // kernel to stall. Therefore, sizing the FIFO is something to consider when
  // designing a 'real' system that uses side channels. In this tutorial, the
  // frequency of updates from the device is so low that this channel can be
  // shallow.
  using MyDeviceToHostSideChannel = 
    DeviceToHostSideChannel<SideChannelPipes::DeviceToHostSideChannelID,
                            int, use_usm_host_alloc, 8>;


  // initialize the side channels
  MyHostToDeviceSideChannel::Init(q);
  MyHostToDeviceTermSideChannel::Init(q);
  MyDeviceToHostSideChannel::Init(q);
  //////////////////////////////////////////////////////////////////////////////

  // get the pointer to the fake input data
  auto i_stream_data = FakeIOPipeInProducer::Data();

  // Create some random input data
  // The range of the random number is [0, count/4), which means the probability
  // of a number ('match_num') matching any number in the list is 1/(count/4)=
  // 4/count. Therefore, the average number of matches in the input stream
  // is count * (4/count) = 4.
  int rand_max = std::max(4, (int)(count / 4));
  size_t frame_size = 1024;
  std::generate_n(i_stream_data, count, [&] { return rand() % rand_max; } );

  // submit the main kernels, once and only once
  auto main_kernel = 
    SubmitSideChannelKernels<ReadIOPipe, WriteIOPipe, MyHostToDeviceSideChannel,
                             MyDeviceToHostSideChannel,
                             MyHostToDeviceTermSideChannel>(q, -1, frame_size);

  //////////////////////////////////////////////////////////////////////////////
  // this lambda will perform a single test to detect all `match_num` elements
  auto test_lambda = [&](int match_num) {
    // determine expected number of updates for this 'match_num'
    size_t expected_updated_count = 
      std::count(i_stream_data, i_stream_data + count, match_num);
    std::vector<int> device_updates;
    device_updates.reserve(expected_updated_count);

    std::cout << "Checking for values matching '" << match_num << "', "
              << "expecting " << expected_updated_count << " matches\n";

    // first, update the kernel with the number to match (blocking)
    MyHostToDeviceSideChannel::write(match_num);

    // This sleep is an artifact of validating the side channel updates.
    // The line of code above writes the new 'match_num' data into
    // a pipe to be read by the main processing kernel on the device. However,
    // just because the value was written to the pipe doesn't mean that the main
    // processing kernel has seen it. For a 'real' streaming design, we
    // typically wouldn't care about this race condition since the host->device
    // updates are asyncrhonous. However, for the sake of this tutorial, we want
    // to be able to validate that the host->device side channel update took
    // place. Therefore, we add a sleep here; 10ms should be plenty of time
    // for the main processing kernel to read the value from the FIFO where we
    // know it resides (since the previous operation is blocking).
    std::this_thread::sleep_for(10ms);

    // launch the producer and consumer to send the data through the kernel
    event producer_dma_event, producer_kernel_event;
    event consumer_dma_event, consumer_kernel_event;
    std::tie(producer_dma_event, producer_kernel_event) =
      FakeIOPipeInProducer::Start(q);
    std::tie(consumer_dma_event, consumer_kernel_event) =
      FakeIOPipeOutConsumer::Start(q);

    // get updates from the device
    for (size_t i = 0; i < expected_updated_count; i++) {
      auto device_update = MyDeviceToHostSideChannel::read();
      device_updates.push_back(device_update);
    }

    // wait for producer and consumer to finish, including the DMA events
    // NOTE: if USM host allocations are used, the dma events are noops.
    producer_dma_event.wait();
    producer_kernel_event.wait();
    consumer_dma_event.wait();
    consumer_kernel_event.wait();

    bool test_passed = true;

    // get the pointer to the fake output data
    auto o_stream_data = FakeIOPipeOutConsumer::Data();

    // validate the output
    for (size_t i = 0; i < count; i++) {
      if (o_stream_data[i] != i_stream_data[i]) {
        std::cerr << "ERROR: output mismatch at entry " << i << ": "
                  << o_stream_data[i] << " != " << i_stream_data[i]
                  << " (out != in)\n";
        test_passed &= false;
      }
    }

    // validate the updates from the device
    for (size_t i = 0; i < expected_updated_count; i++) {
      if (device_updates[i] != match_num) {
        std::cerr << "ERROR: unexpected update value from device: "
                  << device_updates[i] << " != " << match_num
                  << " (update != expected)\n";
        test_passed &= false;
      }
    }

    return test_passed;
  };
  //////////////////////////////////////////////////////////////////////////////

  // run a couple tests with random 'match_num'
  // NOTE: the main processing kernel does NOT exit between these calls
  bool passed = true;
  passed &= test_lambda(rand() % rand_max);
  passed &= test_lambda(rand() % rand_max);
  passed &= test_lambda(rand() % rand_max);

  // we are done testing now, so send a signal to main processing kernel to exit
  MyHostToDeviceTermSideChannel::write(0);

  // wait for the main kernel to finish
  main_kernel.wait();

  // destroy the fake IO pipes and the side channels
  FakeIOPipeInProducer::Destroy(q);
  FakeIOPipeOutConsumer::Destroy(q);
  MyHostToDeviceSideChannel::Destroy(q);
  MyHostToDeviceTermSideChannel::Destroy(q);
  MyDeviceToHostSideChannel::Destroy(q);

  return passed;
}

#endif /* __SIDECHANNELTEST_HPP__ */
