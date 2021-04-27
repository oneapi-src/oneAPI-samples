#define _USE_MATH_DEFINES
#include <cmath>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <thread>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Tuple.hpp"
#include "mvdr_complex.hpp"

#if not defined(REAL_IO_PIPES)
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"
#endif

#include "Constants.hpp"
#include "FakeIOPipes.hpp"
#include "MVDR.hpp"

#if defined(REAL_IO_PIPES) && defined(FPGA_EMULATOR)
static_assert(false, "Real IO pipes cannot be emulated for this design");
#endif

#if defined(REAL_IO_PIPES) && (defined(_WIN32) || defined(_WIN64))
static_assert(false, "Real IO pipes cannot be used in windows");
#endif

#if defined(REAL_IO_PIPES)
#include <sys/mman.h>
#include "UDP.hpp"
#endif

using namespace sycl;
using namespace std::chrono_literals;
using namespace std::chrono;

// We will have the producer and consumer use USM host allocations
// if they are enabled, otherwise they use device allocations
template <typename Id, typename T, size_t min_capacity = 0>
using MyProducer = Producer<Id, T, kUseUSMHostAllocation, min_capacity>;
template <typename Id, typename T, size_t min_capacity = 0>
using MyConsumer = Consumer<Id, T, kUseUSMHostAllocation, min_capacity>;

////////////////////////////////////////////////////////////////////////////////
// utility functions
std::ostream &operator<<(std::ostream &os, const ComplexType &val) {
  os << val.real() << " + " << val.imag() << "i";
  return os;
}

bool AlmostEqual(float x, float y, float epsilon = 0.0002f) {
  return std::fabs(x - y) < epsilon;
}

bool AlmostEqual(ComplexType x, ComplexType y, float epsilon = 0.0002f) {
  bool real_close = AlmostEqual(x.real(), y.real(), epsilon);
  bool imag_close = AlmostEqual(x.imag(), y.imag(), epsilon);
  return real_close && imag_close;
}
////////////////////////////////////////////////////////////////////////////////

// Forward declare the kernel names to reduce name mangling
#if not defined(REAL_IO_PIPES)
class DataProducerID;
class DataOutConsumerID;
#endif
class SinThetaProducerID;

// the data type that goes through the IO pipes; both fake and real
using XrxPipeType = NTuple<ComplexType, kNumComplexPerXrxPipe>;

// size of Training Data matrix, in units of XrxPipeTypes
constexpr size_t kTrainingDataSize =
    kNumSensorInputs * kTrainingMatrixNumRows / kNumComplexPerXrxPipe;

// size of data to be processed, in units of XrxPipeTypes
constexpr size_t kXrxDataSize =
    kNumSensorInputs * kNumInputVectors / kNumComplexPerXrxPipe;

// size of header data per set of training and processing data matrices, in
// units of XrxPipeTypes (one header word per matrix)
constexpr size_t kHeadersSize = 2;

// total size of one 'quanta' of input data, in units of XrxPipeTypes
constexpr size_t kInputDataSize =
    kTrainingDataSize + kXrxDataSize + kHeadersSize;

// total size of one 'quanta' of output data, in units of ComplexTypes
constexpr size_t kDataOutSize = kNumInputVectors * kNumSteer;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// host producer and consumers
#if defined(REAL_IO_PIPES)
// REAL IO PIPES
struct ReadIOPipeID {
  static constexpr unsigned id = 1;
};
struct WriteIOPipeID {
  static constexpr unsigned id = 0;
};

using DataInPipe =
    INTEL::kernel_readable_io_pipe<ReadIOPipeID, XrxPipeType, 512>;

using DataOutPipe =
    INTEL::kernel_writeable_io_pipe<WriteIOPipeID, XrxPipeType, 512>;
#else
// FAKE IO PIPES
using DataProducer =
    MyProducer<DataProducerID, XrxPipeType, kInputDataSize * 2>;
using DataInPipe = DataProducer::Pipe;

using DataOutConsumer =
    MyConsumer<DataOutConsumerID, ComplexType, kDataOutSize * 2>;
using DataOutPipe = DataOutConsumer::Pipe;
#endif

using SinThetaProducer = MyProducer<SinThetaProducerID, float, kNumSteer * 2>;
using SinThetaPipe = SinThetaProducer::Pipe;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// File I/O
bool ReadInputData(std::string in_dir, ComplexType *data_in,
                   int num_matrix_copies);
bool WriteOutputData(std::string out_dir, ComplexType *data_out);
bool CheckOutputData(std::string in_dir, ComplexType *data_out,
                     int num_matrix_copies, bool print_diffs);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
struct UDPArgs {
  unsigned long fpga_mac_addr = 0;
  unsigned long host_mac_addr = 0;
  unsigned int fpga_udp_port = 0;
  unsigned int host_udp_port = 0;
  char *fpga_ip_addr = nullptr;
  char *fpga_netmask = nullptr;
  char *host_ip_addr = nullptr;
};

// arguments
bool ParseArgs(int argc, char *argv[], int &num_matrix_copies,
               std::string &in_dir, std::string &out_dir, UDPArgs *udp_args);
void PrintUsage();
////////////////////////////////////////////////////////////////////////////////

// the main function
int main(int argc, char *argv[]) {
  UDPArgs udp_args;
  int num_matrix_copies = 1024;
  std::string in_dir = "../data";
  std::string out_dir = ".";

  // parse the command line arguments
  if (!ParseArgs(argc, argv, num_matrix_copies, in_dir, out_dir, &udp_args)) {
    PrintUsage();
    std::terminate();
  }

  printf("\n");
#if defined(REAL_IO_PIPES)
  printf("FPGA MAC Address: %012lx\n", udp_args.fpga_mac_addr);
  printf("FPGA IP Address:  %s\n", udp_args.fpga_ip_addr);
  printf("FPGA UDP Port:    %d\n", udp_args.fpga_udp_port);
  printf("FPGA Netmask:     %s\n", udp_args.fpga_netmask);
  printf("Host MAC Address: %012lx\n", udp_args.host_mac_addr);
  printf("Host IP Address:  %s\n", udp_args.host_ip_addr);
  printf("Host UDP Port:    %d\n", udp_args.host_udp_port);
#endif
  printf("Matrices:         %d\n", num_matrix_copies);
  printf("Input Directory:  '%s'\n", in_dir.c_str());
  printf("Output Directory: '%s'\n", out_dir.c_str());
  printf("\n");

  bool passed = true;

  const size_t in_count = kInputDataSize * num_matrix_copies;
  const size_t out_count = kDataOutSize * num_matrix_copies;

  // find number of full matrices.
  // For the real IO pipes, we cannot send and receive a partial
  // packet so we need to find the number of FULL matrices that
  // we will send and receive
#if defined(REAL_IO_PIPES)
  const size_t in_size = in_count * sizeof(ComplexType);

  size_t full_in_packet_count = in_size / kUDPDataSize;
  size_t full_in_size = full_in_packet_count * kUDPDataSize;
  size_t full_in_count = full_in_size / sizeof(ComplexType);

  const size_t num_full_matrix_copies = full_in_count / kInputDataSize;

  const size_t full_out_count = kDataOutSize * num_full_matrix_copies;
  const size_t full_out_size = full_out_count * sizeof(ComplexType);
  const size_t full_out_packet_count = full_out_size / kUDPDataSize;

  std::cout << "full_matrix_copies    = " << num_full_matrix_copies << "\n";
  std::cout << "full_in_packet_count  = " << full_in_packet_count << "\n";
  std::cout << "full_out_packet_count = " << full_out_packet_count << "\n";

  // allocate aligned memory for raw input and output data
  unsigned char *in_packets = AllocatePackets(full_in_packet_count);
  unsigned char *out_packets = AllocatePackets(full_out_packet_count);
#else
  // for the fake IO pipes we don't need to worry about data fitting into
  // UDP packets, so the number of full matrices is the amount requested
  const size_t num_full_matrix_copies = num_matrix_copies;
#endif

  // the input and output data
  std::vector<XrxPipeType> in_data(in_count);
  std::vector<XrxPipeType> out_data(out_count);

  try {
    // device selector
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // create the device queue
#if defined(REAL_IO_PIPES)
    queue q(selector);
#else
    queue q(selector, dpc_common::exception_handler);
#endif

    // initialize the producers and consumers
#if not defined(REAL_IO_PIPES)
    DataProducer::Init(q, kInputDataSize * num_matrix_copies);
    DataOutConsumer::Init(q, kDataOutSize * num_matrix_copies);
#endif
    SinThetaProducer::Init(q, kNumSteer);

    // read the input data
    passed &=
        ReadInputData(in_dir, (ComplexType *)in_data.data(), num_matrix_copies);

#if defined(REAL_IO_PIPES)
    // convert the input data into UDP packets for the real IO pipes
    ToPackets(in_packets, in_data.data(), full_in_count);
#else
    // copy the input data to the producer fake IO pipe buffer
    std::copy_n(in_data.data(), in_count, DataProducer::Data());
#endif

    // calculate the sin(theta) values for each steering vector
    constexpr float degree_unit = 120.0f / (kNumSteer - 1);
    for (int i = 0; i < kNumSteer; i++) {
      float degree = -60.0f + i * degree_unit;
      SinThetaProducer::Data()[i] = sin(degree / 180.0f * M_PI);
    }

    // launch the mvdr kernels
    MVDREventArray mvdr_events;
    mvdr_events = SubmitMVDRKernels<
        kNumSensorInputs,           // number of sensor array inputs
        kRMBFactor,                 // Reed-Mallett-Brennan rule
                                    // Number of 'rows' of sensor data used
                                    // by the QRD is k_num_sensor_inputs *
                                    // k_rmb_factor (generally 2-5)
        kNumSteer,                  // number of steering vectors
        kSubstitutionUnrollFactor,  // unroll factor used by the forward and
                                    // backward substitution kernels
        kBeamformingUnrollFactor,   // unroll factor used by beamformer
        kQRDMinIterations,          // minumum 'inner loop' iterations for QRD
        kNumComplexPerXrxPipe,      // Number of complex numbers (contained
                                    // in NTuple) per read from the
                                    // Xrx input pipes
        DataInPipe,                 // Input data for MVDR
        SinThetaPipe,               // sin(theta) input for steering vectors
        DataOutPipe                 // output from MVDR
        >(q, kNumInputVectors);

    std::cout << "Launched MVDR kernels" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // Throughput test
    ////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl
              << "*** Launching throughput test of " << num_matrix_copies
              << " matrices ***" << std::endl;

    std::cout << "Sensor inputs                 : " << kNumSensorInputs
              << std::endl;
    std::cout << "Training matrix rows          : " << kTrainingMatrixNumRows
              << std::endl;
    std::cout << "Data rows per training matrix : " << kNumInputVectors
              << std::endl;
    std::cout << "Steering vectors              : " << kNumSteer << std::endl;
    std::cout << "Streaming pipe width          : " << kNumComplexPerXrxPipe
              << std::endl;

    // Start the host producer for the sin theta data
    // By waiting on this kernel to finish, we are assuring that the runtime
    // has already programmed the FPGA with the image for this kernel. This
    // makes calling SetupPAC below safe (for the real IO pipes).
    event steer_dma_event, steer_kernel_event;
    std::tie(steer_dma_event, steer_kernel_event) =
      SinThetaProducer::Start(q, kNumSteer);
    steer_dma_event.wait();
    steer_kernel_event.wait();

    // Setup CSRs on FPGA (this function is in UDP.hpp)
    // NOTE: this must be done AFTER programming the FPGA, which happens
    // when the kernel is launched (if it is not already programmed)
#if defined(REAL_IO_PIPES)
    SetupPAC(udp_args.fpga_mac_addr, udp_args.fpga_ip_addr,
             udp_args.fpga_udp_port, udp_args.fpga_netmask,
             udp_args.host_mac_addr, udp_args.host_ip_addr,
             udp_args.host_udp_port);

    // give some time for the FPGA to set things up
    std::this_thread::sleep_for(100ms);
#endif

    ////////////////////////////////////////////////////////////////////////////
    // start the producer and consumer of data
#if defined(REAL_IO_PIPES)
    // REAL IO PIPES: start CPU threads to produce/consume data to/from the
    // IO pipes of the FPGA using UDP

    // start the receiver
    std::cout << "Starting receiver thread" << std::endl;
    std::thread receiver_thread([&] {
      std::this_thread::sleep_for(10ms);
      std::cout << "Receiver running on CPU " << sched_getcpu() << "\n";
      UDPReceiver(udp_args.host_ip_addr, udp_args.fpga_udp_port, out_packets,
                  full_out_packet_count, nullptr);
    });

    // pin the receiver to CPU 1
    if (PinThreadToCPU(receiver_thread, 1) != 0) {
      std::cerr << "ERROR: could not pin receiver thread to core 1\n";
    }

    // give a little time for the receiver to start
    std::this_thread::sleep_for(20ms);

    // Start the sender
    std::cout << "Starting sender thread" << std::endl;
    std::thread sender_thread([&] {
      std::this_thread::sleep_for(10ms);
      std::cout << "Sender running on CPU " << sched_getcpu() << "\n";
      UDPSender(udp_args.fpga_ip_addr, udp_args.host_udp_port, in_packets,
                full_in_packet_count, nullptr);
    });

    // pin the sender to CPU 3
    if (PinThreadToCPU(sender_thread, 3) != 0) {
      std::cerr << "ERROR: could not pin sender thread to core 3\n";
    }
#else
    // start the fake IO pipe kernels
    event consume_dma_event, consume_kernel_event;
    event produce_dma_event, produce_kernel_event;
    std::tie(consume_dma_event, consume_kernel_event) =
        DataOutConsumer::Start(q, kDataOutSize * num_matrix_copies);
    std::tie(produce_dma_event, produce_kernel_event) =
        DataProducer::Start(q, kInputDataSize * num_matrix_copies);
#endif
    ////////////////////////////////////////////////////////////////////////////

    // Wait for the DMA event to finish for the producer before starting the
    // timer. If USM host allocations are used, this is a noop.
    produce_dma_event.wait();

    auto start_time = high_resolution_clock::now();

    // wait for producer and consumer to finish
#if defined(REAL_IO_PIPES)
    sender_thread.join();
    receiver_thread.join();
#else
    produce_kernel_event.wait();
    consume_kernel_event.wait();
#endif

    auto end_time = high_resolution_clock::now();

    // Stop the timer before performing the DMA from the consumer. Again,
    // if USM host allocations are used then this is a noop.
    consume_dma_event.wait();

    // compute latency and throughput
    duration<double, std::milli> process_time(end_time - start_time);
    double latency_s = process_time.count() * 1e-3;
    double throughput = num_full_matrix_copies / latency_s;

    std::cout << "Throughput: " << throughput << " matrices/second\n";

    // copy the output back from the consumer
#if defined(REAL_IO_PIPES)
    const size_t count_to_extract =
        full_out_packet_count * kUDPDataSize / sizeof(ComplexType);

    FromPackets(out_packets, out_data.data(), count_to_extract);

    const size_t num_out_matrix_copies_to_check =
        count_to_extract / kDataOutSize;
#else
    std::copy_n(DataOutConsumer::Data(), out_count,
                (ComplexType *)out_data.data());

    const size_t num_out_matrix_copies_to_check = num_full_matrix_copies;
#endif

    // check one instance of output data
    passed &= CheckOutputData(in_dir, (ComplexType *)out_data.data(),
                              num_out_matrix_copies_to_check, true);
    if (passed) {
      std::cout << "Output data check succeeded" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Post-test cleanup
    ////////////////////////////////////////////////////////////////////////////
#if defined(REAL_IO_PIPES)
    FreePackets(in_packets, full_in_packet_count);
    FreePackets(out_packets, full_out_packet_count);
#else
    DataProducer::Destroy(q);
    DataOutConsumer::Destroy(q);
#endif
    SinThetaProducer::Destroy(q);

  } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

bool ReadInputData(std::string in_dir, ComplexType *data_in,
                   int num_matrix_copies) {
  // file paths relative the the base directory
  std::string training_real_path = in_dir + "/" + "A_real.txt";
  std::string training_imag_path = in_dir + "/" + "A_imag.txt";
  std::string x_real_path = in_dir + "/" + "X_real.txt";
  std::string x_imag_path = in_dir + "/" + "X_imag.txt";

  std::cout << "Reading training data from '" << training_real_path << " and "
            << training_imag_path << std::endl;

  // parse A
  std::ifstream a_real_is, a_imag_is;
  a_real_is.open(training_real_path);
  a_imag_is.open(training_imag_path);
  if (a_real_is.fail()) {
    std::cerr << "Failed to open " << training_real_path << "\n";
    return false;
  }
  if (a_imag_is.fail()) {
    std::cerr << "Failed to open " << training_imag_path << "\n";
    return false;
  }

  constexpr float kIsTrainingData = 0.0f;
  constexpr float kIsNotTrainingData = 1.0f;  // any non-zero number is fine

  // insert the header to mark the first training matrix
  data_in[0].real() = std::nanf("");    // marks this word as a header
  data_in[0].imag() = kIsTrainingData;  // marks this as training data

  // load the first matrix from the input file
  int data_offset = kNumComplexPerXrxPipe;  // skip the header
  for (size_t i = 0; i < kTrainingMatrixNumRows * kNumSensorInputs; i++) {
    a_real_is >> data_in[data_offset + i].real();
    a_imag_is >> data_in[data_offset + i].imag();
  }

  a_real_is.close();
  a_imag_is.close();

  std::cout << "Reading input data from " << x_real_path << " and "
            << x_imag_path << std::endl;

  // parse X
  std::ifstream x_real_is, x_imag_is;
  x_real_is.open(x_real_path);
  x_imag_is.open(x_imag_path);
  if (x_real_is.fail()) {
    std::cerr << "Failed to open " << x_real_path << "\n";
    return false;
  }
  if (x_imag_is.fail()) {
    std::cerr << "Failed to open " << x_imag_path << "\n";
    return false;
  }

  // insert header to mark processing data
  data_offset = (kTrainingDataSize + 1) * kNumComplexPerXrxPipe;
  data_in[data_offset].real() = std::nanf("");
  data_in[data_offset].imag() = kIsNotTrainingData;
  data_offset += kNumComplexPerXrxPipe;

  // load the first matrix
  for (size_t i = 0; i < kInputDataSize * kNumComplexPerXrxPipe; i++) {
    x_real_is >> data_in[data_offset + i].real();
    x_imag_is >> data_in[data_offset + i].imag();
  }

  x_real_is.close();
  x_imag_is.close();

  // copy the first data and training matrices num_matrix_copies times
  for (size_t matrix_num = 1; matrix_num < num_matrix_copies; matrix_num++) {
    for (size_t i = 0; i < kInputDataSize * kNumComplexPerXrxPipe; i++) {
      size_t data_copy_index =
          i + (matrix_num * kInputDataSize * kNumComplexPerXrxPipe);
      data_in[data_copy_index] = data_in[i];
    }
  }

  return true;
}

bool CheckOutputData(std::string in_dir, ComplexType *data_out,
                     int num_matrix_copies, bool print_diffs) {
  bool match = true;

  // file paths relative the the base directory
  std::string expected_out_real_path =
      in_dir + "/" + "small_expected_out_real.txt";
  std::string expected_out_imag_path =
      in_dir + "/" + "small_expected_out_imag.txt";
#ifdef LARGE_SENSOR_ARRAY
  expected_out_real_path = in_dir + "/" + "large_expected_out_real.txt";
  expected_out_imag_path = in_dir + "/" + "large_expected_out_imag.txt";
#endif
  if (print_diffs) {
    std::cout << "Checking output data against " << expected_out_real_path
              << " and " << expected_out_imag_path << std::endl;
  }

  std::ifstream exp_real_is, exp_imag_is;
  exp_real_is.open(expected_out_real_path);
  exp_imag_is.open(expected_out_imag_path);
  if (exp_real_is.fail()) {
    std::cerr << "Failed to open " << expected_out_real_path << "\n";
    return false;
  }
  if (exp_imag_is.fail()) {
    std::cerr << "Failed to open " << expected_out_imag_path << "\n";
    return false;
  }

  // parse the expected output data
  std::vector<ComplexType> ref_data(kNumInputVectors * kNumSteer);
  for (size_t i = 0; i < kNumInputVectors; i++) {
    for (size_t j = 0; j < kNumSteer; j++) {
      exp_real_is >> ref_data[i * kNumSteer + j].real();
      exp_imag_is >> ref_data[i * kNumSteer + j].imag();
    }
  }

  exp_real_is.close();
  exp_imag_is.close();

  // validate the result for all output matrices
  for (size_t m = 0; m < num_matrix_copies; m++) {
    for (size_t i = 0; i < kNumInputVectors; i++) {
      for (size_t j = 0; j < kNumSteer; j++) {
        auto result =
            data_out[m * kNumSteer * kNumInputVectors + i * kNumSteer + j];
        auto expected = ref_data[i * kNumSteer + j];

        if (!AlmostEqual(expected, result)) {
          if (print_diffs) {
            std::cout << "Error in output matrix " << m << " for input vector "
                      << i << ", steering vector " << j << ".\n"
                      << "Expected: " << expected << ", "
                      << "Actual: " << result << "\n";
          }
          match = false;
        }
      }
    }
  }

  return match;
}

bool WriteOutputData(std::string out_dir, ComplexType *data_out) {
  // file paths relative the the base directory
  std::string out_real_path = out_dir + "/" + "out_real.txt";
  std::string out_imag_path = out_dir + "/" + "out_imag.txt";

  std::cout << "Writing output to " << out_real_path << " and " << out_imag_path
            << std::endl;

  std::ofstream out_real_os, out_imag_os;
  out_real_os.open(out_real_path, std::ios::out | std::ios::trunc);
  out_imag_os.open(out_imag_path, std::ios::out | std::ios::trunc);
  if (out_real_os.fail()) {
    std::cerr << "Failed to open " << out_real_path << "\n";
    return false;
  }
  if (out_imag_os.fail()) {
    std::cerr << "Failed to open " << out_imag_path << "\n";
    return false;
  }

  for (size_t i = 0; i < kNumInputVectors; i++) {
    for (size_t j = 0; j < kNumSteer; j++) {
      out_real_os << std::fixed << std::setw(11)
                  << data_out[i * kNumSteer + j].real() << " ";
      out_imag_os << std::fixed << std::setw(11)
                  << data_out[i * kNumSteer + j].imag() << " ";
    }
    out_real_os << std::endl;
    out_imag_os << std::endl;
  }

  out_real_os.close();
  out_imag_os.close();

  return true;
}

bool ParseArgs(int argc, char *argv[], int &num_matrix_copies,
               std::string &in_dir, std::string &out_dir, UDPArgs *udp_args) {
#if defined(REAL_IO_PIPES)
  if (argc < 8) {
    return false;
  } else {
    // parse FPGA and HOST MAC addresses
    udp_args->fpga_mac_addr = ParseMACAddress(argv[1]);
    udp_args->host_mac_addr = ParseMACAddress(argv[5]);

    // parse ports
    udp_args->fpga_udp_port = (unsigned int)atoi(argv[3]);
    udp_args->host_udp_port = (unsigned int)atoi(argv[7]);

    // get IP addresses and netmask
    udp_args->fpga_ip_addr = argv[2];
    udp_args->fpga_netmask = argv[4];
    udp_args->host_ip_addr = argv[6];

    if (argc > 8) {
      if (atoi(argv[8]) > 0) {
        num_matrix_copies = atoi(argv[8]);
      }
    }
    if (argc > 9) {
      in_dir = argv[9];
    }
    if (argc > 10) {
      out_dir = argv[10];
    }

    return true;
  }
#else
  if (argc > 1) {
    if (atoi(argv[1]) > 0) {
      num_matrix_copies = atoi(argv[1]);
    }
  }
  if (argc > 2) {
    in_dir = argv[2];
  }
  if (argc > 3) {
    out_dir = argv[3];
  }
  return true;
#endif
}

void PrintUsage() {
#if defined(REAL_IO_PIPES)
  std::cout << "USAGE: ./mvdr_beamforming.fpga fpga_mac_adr fpga_ip_addr "
            << "fpga_udp_port fpga_netmask host_mac_adr host_ip_addr "
            << "host_udp_port [num_matrices] [in directory] [out directory]\n";
  std::cout << "EXAMPLE: ./mvdr_beamforming.fpga 64:4C:36:00:2F:20 "
            << "192.168.0.11 34543 255.255.255.0 94:40:C9:71:8D:10 "
            << " 192.168.0.10 34543 1024 ../data .\n";
#else
  std::cout << "USAGE: ./mvdr_beamforming.fpga "
            << "[num_matrices] [in directory] [out directory]\n";
  std::cout << "EXAMPLE: ./mvdr_beamforming.fpga 1024 ../data .\n";
#endif
}
