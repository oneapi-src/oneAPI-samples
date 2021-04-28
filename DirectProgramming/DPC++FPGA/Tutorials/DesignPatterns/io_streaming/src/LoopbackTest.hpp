#ifndef __LOOPBACKTEST_HPP__
#define __LOOPBACKTEST_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "FakeIOPipes.hpp"

// If the 'USE_REAL_IO_PIPE' macro is defined, this test will use real IO pipes.
// To use this, ensure you have a BSP that supports IO pipes.
// NOTE: define this BEFORE including the LoopbackTest.hpp and
// SideChannelTest.hpp which will check for the presence of this macro.
//#define USE_REAL_IO_PIPES

using namespace sycl;

// declare the kernel and pipe ID stucts globally to reduce name mangling
struct LoopBackMainKernel;
struct LoopBackIOPipes {
  struct ReadIOPipeID { static constexpr unsigned id = 0; };
  struct WriteIOPipeID { static constexpr unsigned id = 1; };
};

//
// The simplest processing kernel. Streams data in 'IOPipeIn' and streams
// it out 'IOPipeOut'. The developer of this kernel uses this abstraction
// to create a streaming kernel. They don't particularly care whether the IO
// pipes are 'real' or not, that is up to the system designer who works on
// stitching together the whole system. In this tutorial, the stitching of the
// full system is done below in the 'RunLoopbackSystem' function.
//
template<class IOPipeIn, class IOPipeOut>
event SubmitLoopbackKernel(queue& q, size_t count) {
  return q.submit([&](handler& h) {
    h.single_task<LoopBackMainKernel>([=] {
      for (size_t i = 0; i < count; i++) {
        auto data = IOPipeIn::read();
        // !!! Your processing can go here !!!
        IOPipeOut::write(data);
      }
    });
  });
}

//
// Run the loopback system
//
template<typename T, bool use_usm_host_alloc>
bool RunLoopbackSystem(queue& q, size_t count) {
  bool passed = true;

  //////////////////////////////////////////////////////////////////////////////
  // IO pipes
  constexpr size_t kIOPipeDepth = 4;
#ifndef USE_REAL_IO_PIPES
  // these are FAKE IO pipes (and their producer/consumer)
  using FakeIOPipeInProducer = Producer<LoopBackIOPipes::ReadIOPipeID,
                                T, use_usm_host_alloc, kIOPipeDepth>;
  using FakeIOPipeOutConsumer = Consumer<LoopBackIOPipes::WriteIOPipeID,
                                 T, use_usm_host_alloc, kIOPipeDepth>;
  using ReadIOPipe = typename FakeIOPipeInProducer::Pipe;
  using WriteIOPipe = typename FakeIOPipeOutConsumer::Pipe;

  // initialize the fake IO pipes
  FakeIOPipeInProducer::Init(q, count);
  FakeIOPipeOutConsumer::Init(q, count);
#else
  // these are REAL IO pipes
  using ReadIOPipe = 
    INTEL::kernel_readable_io_pipe<LoopBackIOPipes::ReadIOPipeID,
                                   T, kIOPipeDepth>;
  using WriteIOPipe =
    INTEL::kernel_writeable_io_pipe<LoopBackIOPipes::WriteIOPipeID,
                                    T, kIOPipeDepth>;
#endif
  //////////////////////////////////////////////////////////////////////////////

  // FAKE IO PIPES ONLY
#ifndef USE_REAL_IO_PIPES
  // get the pointer to the fake input data
  auto i_stream_data = FakeIOPipeInProducer::Data();

  // create some random input data for the fake IO pipe
  std::generate_n(i_stream_data, count, [&] { return rand() % 100; } );
#endif

  // submit the main processing kernel
  auto kernel_event = SubmitLoopbackKernel<ReadIOPipe, WriteIOPipe>(q, count);

  // FAKE IO PIPES ONLY
#ifndef USE_REAL_IO_PIPES
  // start the producer and consumer
  event produce_dma_e, produce_kernel_e;
  event consume_dma_e, consume_kernel_e;
  std::tie(produce_dma_e, produce_kernel_e) = FakeIOPipeInProducer::Start(q);
  std::tie(consume_dma_e, consume_kernel_e) = FakeIOPipeOutConsumer::Start(q);

  // wait for producer and consumer to finish including the DMA events.
  // NOTE: if USM host allocations are used, the dma events are noops.
  produce_dma_e.wait();
  produce_kernel_e.wait();
  consume_dma_e.wait();
  consume_kernel_e.wait();
#endif

  // Wait for main kernel to finish.
  // NOTE: we can only wait on the loopback kernel because it knows how much
  // data it expects to process ('count'). In general, this may not be the
  // case and you may want the processing kernel to run 'forever' (or until the
  // host tells it to stop). For an example of this, see 'SideChannelTest.hpp'.
  kernel_event.wait();

  // FAKE IO PIPES ONLY
#ifndef USE_REAL_IO_PIPES
  // get the pointer to the fake input data
  auto o_stream_data = FakeIOPipeOutConsumer::Data();

  // validate the output
  for (size_t i = 0; i < count; i++) {
    if (o_stream_data[i] != i_stream_data[i]) {
      std::cerr << "ERROR: output mismatch at entry " << i << ": "
                << o_stream_data[i] << " != " << i_stream_data[i]
                << " (out != in)\n";
      passed &= false;
    }
  }
#endif

  return passed;
}


#endif /* __LOOPBACKTEST_HPP__ */
