#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>

#include <sys/mman.h>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/ac_types/ac_complex.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Tuple.hpp"
#include "UDP.hpp"

using WordType = unsigned long long;
static_assert(sizeof(WordType) <= kUDPDataSize);
static_assert((kUDPDataSize % sizeof(WordType)) == 0);
constexpr size_t kWordsPerPacket = kUDPDataSize / sizeof(WordType);

using namespace sycl;
using namespace std::chrono;

// declare kernel and pipe IDs globally to reduce name mangling
class RealLoopbackKernel;
class NoopKernel;
struct WriteIOPipeID {
  static constexpr unsigned id = 0;
};
struct ReadIOPipeID {
  static constexpr unsigned id = 1;
};

// the IO pipes
struct IOPipeType {
  uchar dat[8];
};
using WriteIOPipe =
    INTEL::kernel_writeable_io_pipe<WriteIOPipeID, IOPipeType, 512>;
using ReadIOPipe =
    INTEL::kernel_readable_io_pipe<ReadIOPipeID, IOPipeType, 512>;

// submits the main processing kernel, templated on the input and output pipes
// NOTE: this kernel is agnostic to whether the input/output pipe is a 'real'
// or 'fake' IO pipe.
template <typename IOPipeIn, typename IOPipeOut>
event SubmitLoopbackKernel(queue &q) {
  return q.submit([&](handler &h) {
    h.single_task<RealLoopbackKernel>([=] {
      while (1) {
        auto data = IOPipeIn::read();
        IOPipeOut::write(data);
      }
    });
  });
}

// this is a kernel that does no work, it is used as a detection mechanism for
// the FPGA being programmed
event SubmitNoopKernel(queue &q) {
  return q.submit([&](handler &h) { h.single_task<NoopKernel>([=] {}); });
}

int main(int argc, char **argv) {
  unsigned long fpga_mac_adr = 0;
  unsigned long host_mac_adr = 0;
  unsigned int fpga_udp_port = 0;
  unsigned int host_udp_port = 0;
  char *fpga_ip_address = nullptr;
  char *fpga_netmask = nullptr;
  char *host_ip_address = nullptr;
  size_t packets = 100000000;

  if (argc < 8) {
    std::cout << "USAGE: ./mvdr_beamforming.fpga fpga_mac_adr fpga_ip_addr "
              << "fpga_udp_port fpga_netmask host_mac_adr host_ip_addr "
              << "host_udp_port [packets]\n";
    std::cout << "EXAMPLE: ./mvdr_beamforming.fpga 64:4C:36:00:2F:20 "
              << "192.168.0.11 34543 255.255.255.0 94:40:C9:71:8D:10 "
              << " 192.168.0.10 34543 1024 .\n";
  }

  // parse FPGA and HOST MAC addresses
  fpga_mac_adr = ParseMACAddress(argv[1]);
  host_mac_adr = ParseMACAddress(argv[5]);

  // parse ports
  fpga_udp_port = (unsigned int)atoi(argv[3]);
  host_udp_port = (unsigned int)atoi(argv[7]);

  // get IP addresses and netmask
  fpga_ip_address = argv[2];
  fpga_netmask = argv[4];
  host_ip_address = argv[6];

  // parse number of packets (optional arg)
  if (argc > 8) {
    packets = atoi(argv[8]);
  }

  printf("\n");
  printf("FPGA MAC Address: %012lx\n", fpga_mac_adr);
  printf("FPGA IP Address:  %s\n", fpga_ip_address);
  printf("FPGA UDP Port:    %d\n", fpga_udp_port);
  printf("FPGA Netmask:     %s\n", fpga_netmask);
  printf("Host MAC Address: %012lx\n", host_mac_adr);
  printf("Host IP Address:  %s\n", host_ip_address);
  printf("Host UDP Port:    %d\n", host_udp_port);
  printf("Packets:          %zu\n", packets);
  printf("\n");

  // set up SYCL queue
  INTEL::fpga_selector device_selector;
  queue q(device_selector);

  // input and output packet words
  const size_t total_elements = kWordsPerPacket * packets;
  std::vector<WordType> input(total_elements);
  std::vector<WordType> output(total_elements);

  // fill input, set output
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  // allocate aligned memory for input and output data
  // total bytes including the 2-byte header per packet
  unsigned char *input_data = AllocatePackets(packets);
  unsigned char *output_data = AllocatePackets(packets);

  // prepare data to be transferred, added check header 0xABCD, reset output
  std::cout << "Creating packets from input data" << std::endl;
  ToPackets(input_data, input.data(), total_elements);

  // these are used to track the latency of each packet
  std::vector<high_resolution_clock::time_point> time_in(packets);
  std::vector<high_resolution_clock::time_point> time_out(packets);

  // start the main kernel
  std::cout << "Starting processing kernel on the FPGA" << std::endl;
  auto kernel_event = SubmitLoopbackKernel<ReadIOPipe, WriteIOPipe>(q);

  // Here, we start the noop kernel and wait for it to finish.
  // By waiting on the finish we assure that the FPGA has been programmed with
  // the image and the setting of CSRs SetupPAC will not be undone with by the
  // runtime reprogramming the FPGA.
  std::cout << "Starting and waiting on Noop kernel to ensure "
            << "that the FPGA is programmed \n";
  SubmitNoopKernel(q).wait();

  // Setup CSRs on FPGA (this function is in UDP.hpp)
  // NOTE: this must be done AFTER programming the FPGA, which happens
  // when the kernel is launched (if it is not already programmed)
  SetupPAC(fpga_mac_adr, fpga_ip_address, fpga_udp_port, fpga_netmask,
           host_mac_adr, host_ip_address, host_udp_port);
  std::this_thread::sleep_for(10ms);

  // Start the receiver
  std::cout << "Starting receiver thread" << std::endl;
  std::thread receiver_thread([&] {
    std::this_thread::sleep_for(20ms);
    std::cout << "Receiver running on CPU " << sched_getcpu() << "\n";
    UDPReceiver(host_ip_address, fpga_udp_port, output_data, packets,
                time_out.data());
  });

  // pin the receiver to core 1
  if (PinThreadToCPU(receiver_thread, 1) != 0) {
    std::cerr << "ERROR: could not pin receiver thread to core 1\n";
  }

  // give some time for the receiever to start
  std::this_thread::sleep_for(10ms);

  // Start the sender
  std::cout << "Starting sender thread" << std::endl;
  std::thread sender_thread([&] {
    std::this_thread::sleep_for(20ms);
    std::cout << "Sender running on CPU " << sched_getcpu() << "\n";
    UDPSender(fpga_ip_address, host_udp_port, input_data, packets,
              time_in.data());
  });

  // pin the sender to core 3
  if (PinThreadToCPU(sender_thread, 3) != 0) {
    std::cerr << "ERROR: could not pin sender thread to core 3\n";
  }

  // wait for sender and receiver threads to finish
  sender_thread.join();
  receiver_thread.join();

  // DON'T wait for kernel to finish, since it is an infinite loop
  // kernel_event.wait();

  // compute average latency (ignore the first couple of packets)
  double avg_latency = 0.0;
  constexpr size_t kWarmupPackets = 8;
  for (size_t i = kWarmupPackets; i < packets; i++) {
    duration<double, std::milli> diff(time_out[i] - time_in[i]);
    avg_latency += diff.count();
  }
  avg_latency /= (packets - kWarmupPackets);
  std::cout << "Average end-to-end packet latency: " << avg_latency << " ms\n";

  // convert the output bytes to data (drop the headers)
  std::cout << "Getting output data from packets" << std::endl;
  FromPackets(output_data, output.data(), total_elements);

  // validate results
  bool passed = true;
  for (size_t i = 0; i < total_elements; i++) {
    if (output[i] != input[i]) {
      std::cerr << "ERROR: output mismatch at index " << i << ", " << output[i]
                << " != " << input[i] << " (output != input)\n";
      passed = false;
    }
  }

  // Custom free function unpins and frees memory
  FreePackets(input_data, packets);
  FreePackets(output_data, packets);

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}
