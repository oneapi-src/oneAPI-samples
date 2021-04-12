// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "Constants.hpp"
#include "FakeIOPipes.hpp"
#include "SteeringVectorGenerator.hpp"

using namespace sycl;

// utility functions
void PrintComplex(const ComplexType& c) {
  std::cout << "(" << c.real() << ", " << c.imag() << ")";
}

bool AlmostEqual(float x, float y, float epsilon = 0.0001f) {
  return std::fabs(x - y) < epsilon;
}

bool AlmostEqual(ComplexType x, ComplexType y, float epsilon = 0.0001f) {
  bool real_close = AlmostEqual(x.real(), y.real(), epsilon);
  bool imag_close = AlmostEqual(x.imag(), y.imag(), epsilon);
  return real_close && imag_close;
}


// File I/O
bool ReadComplexData(std::string file_basename, ComplexType *data);

// forward declare kernel and pipe names
class SinThetaProducerID;
class SinThetaProducerPipeID;
class SteeringVectorConsumerID;
class SteeringVectorConsumerPipeID;
class SteeringVectorGenerator;
class UpdateSinThetaPipeID;
class UpdateSteeringVectorsPipeID;
class UpdateSinThetaKernelName;



// the main function
int main(int argc, char *argv[]) {
  bool need_help = false;
  std::string in_dir = "../data";
  std::string out_dir = "../data";

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      need_help = true;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--in=") == 0) {
        in_dir = std::string(str_after_equals.c_str());
      } else if (arg.find("--out=") == 0) {
        out_dir = std::string(str_after_equals.c_str());
      } else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // print help if needed
  if (need_help) {
    std::cout << "USAGE: ./mvdr_kernel_tests "
              << "[--in=<string>] "
              << "[--out=<string>\n";
    return 0;
  }

  bool passed = true;

  // start the more 'SYCL' portion of the design
  try {
    // device selector
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // queue properties to enable SYCL profiling of kernels
    auto prop_list = property_list{ property::queue::enable_profiling() };

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // make sure the device supports USM host and device allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      return 1;
    }
    if (!d.get_info<info::device::usm_device_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM device"
                << " allocations\n";
      return 1;
    }

    // allocate host memory
    float *sintheta_in;
    if( (sintheta_in = malloc_host<float>( kNumSteer, q) ) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'sintheta_in'\n";
      std::terminate();
    }
    ComplexType *steering_vectors_out;
    constexpr short kSteeringVectorsSize = kNumSteer * kNumSensorInputs;
    if( (steering_vectors_out = 
      malloc_host<ComplexType>( kSteeringVectorsSize, q) ) == nullptr )
    {
      std::cerr << "ERROR: could not allocate space for "
      std::cerr << "'steering_vectors_out'\n";
      std::terminate();
    }

    // clear the output data
    std::fill_n(steering_vectors_out, kSteeringVectorsSize, -1);

    // read the input data
    std::string sintheta_in_base_filename = indir + "/sintheta_in"
    passed &= ReadFloatData( sintheta_in_base_filename, sintheta_in );

    // Declare pipes, producers, and consumers
    using SinThetaProducer = HostProducer<  SinThetaProducerID,
                                            SinThetaProducerPipeID,
                                            float >;
    using SinThetaPipe = SinThetaProducer::Pipe;

    using UpdateSinThetaPipe = sycl::pipe< UpdateSinThetaPipeID, bool, 1 >;

    using SteeringVectorsConsumer = HostConsumer< SteeringVectorConsumerID,
                                                  SteeringVectorConsumerPipeID,
                                                  ComplexType >;
    using SteeringVectorsPipe = SteeringVectorsConsumer::Pipe;

    using UpdateSteeringVectorsPipe = 
      sycl::pipe< UpdateSteeringVectorsPipeID, bool, 1 >;

    SinThetaProducer sintheta_producer(q, sintheta_in);
    SteeringVectorsConsumer steering_vectors_consumer(q, steering_vectors_out);

    // start the steering vector generator kernel
    event steering_vector_generator_event =
      SubmitSteeringVectorGeneratorKernel<
        SteeringVectorGenerator,        // Name to use for the Kernel
        kNumSteer,                      // number of steering vectors
        kNumSensorInputs,               // number of elements in each vector
        SinThetaPipe,                   // sin(theta) input
        UpdateSinThetaPipe,             // load new sin(theta)
        SteeringVectorsPipes,           // generated steering vectors
        UpdateSteeringVectorsPipes      // load new steering vectors
      >( q );

    // start the Producer and Consumer kernels
    event steering_vector_consumer_event = 
      steering_vectors_consumer.Start(kSteeringVectorsSize);
    event sintheta_producer_event = sintheta_producer.Start(kNumSteer);

    // wait for the producer kernel to finish before writing to the 
    // UpdateSinTheta pipe
    sintheta_producer_event.wait();

    // launch kernel that writes a single bool to the UpdateSinTheta pipe
    event update_sin_theta_event = q.submit([&](handler& h) {
      h.single_task< UpdateSinThetaKernelName > ( [=]() {
        UpdateSinThetaPipe::write( true );
      });
    });
    update_sin_theta_event.wait();

    // wait for the kernel and consumer to finish
    steering_vector_generator_event.wait();
    steering_vector_consumer_event.wait();

    // print the output
    for ( int i = 0; i < kNumSteer; i++ ) {
      std::cout << std::endl << "Vector " << i << std::endl;
      for ( int j = 0; j < kNumSensorInputs; j++ ) {
        PrintComplex( steering_vectors_out[i * kNumSensorInputs + j]);
        std::cout << std::endl;
      }
    }

    // free the host USM memory
    sycl::free(sintheta_in, q);
    sycl::free(steering_vectors_out, q);

  } catch (exception const& e) {
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
    std::cout << std::endl << "PASSED\n";
    return 0;
  } else {
    std::cout << std::endl << "FAILED\n";
    return 1;
  }
}

bool ReadInputData(std::string in_dir, ComplexType *a, ComplexType *x) {
  std::cout << "Reading input from '" << in_dir << "'\n";

  // file paths relative the the base directory
  std::string a_real_path = in_dir + "/" + "A_real.txt";
  std::string a_imag_path = in_dir + "/" + "A_imag.txt";
  std::string x_real_path = in_dir + "/" + "X_real.txt";
  std::string x_imag_path = in_dir + "/" + "X_imag.txt";

  ComplexBaseType real, imag;

  // parse A
  // TODO: allow multiple input matrices back-to-back
  std::ifstream a_real_is, a_image_is;
  a_real_is.open(a_real_path);
  a_image_is.open(a_imag_path); 
  if (a_real_is.fail()) {
    std::cerr << "Failed to open " << a_real_path << "\n";
    return false;
  }
  if (a_image_is.fail()) {
    std::cerr << "Failed to open " << a_imag_path << "\n";
    return false;
  }

  for (size_t i = 0; i < kASize; i++) {
    a_real_is >> real;
    a_image_is >> imag;
    a[i] = ComplexType(real, imag);
  }

  a_real_is.close();
  a_image_is.close();

  // parse X
  std::ifstream x_real_is, x_image_is;
  x_real_is.open(x_real_path);
  x_image_is.open(x_imag_path); 
  if (x_real_is.fail()) {
    std::cerr << "Failed to open " << x_real_path << "\n";
    return false;
  }
  if (x_image_is.fail()) {
    std::cerr << "Failed to open " << x_imag_path << "\n";
    return false;
  }

  for (size_t i = 0; i < kXSize; i++) {
    x_real_is >> real;
    x_image_is >> imag;
    x[i] = ComplexType(real, imag);
  }

  x_real_is.close();
  x_image_is.close();

  return true;
}

// test the conversion to and from UDP data
// This is the layout of the test pipeline:
//
// |------------|  |---------| InputPipe  |----| OutputPipe  |---------|  |------------|
// |HostProducer|=>|UDPReader|===========>|Wire|============>|UDPWriter|=>|HostConsumer|
// |------------|  |---------|            |----|             |---------|  |------------|
//
template<typename T>
void UDPDataConverterTest(queue& q, T* in_data, size_t count) {
  // number of data bytes from UDP
  constexpr size_t udp_bytes = 8;

  // the UDP data type
  using UDP_t = UDPData<udp_bytes>;

  // number of application elements per UDP packet
  constexpr size_t elements_per_udp_packet = udp_bytes / sizeof(T);
  static_assert(elements_per_udp_packet == 1,
      "We want 1 complex number per UDP packet");
  size_t num_packets = (count + elements_per_udp_packet - 1)  / elements_per_udp_packet;

  // input and output pipes for application (before/after UDP conversion)
  using InputPipe = sycl::pipe<class PipeIn, T>;
  using OutputPipe = sycl::pipe<class PipeOut, T>;

  // allocate fake UDP data
  UDP_t *in, *out;
  if((in = malloc_host<UDP_t>(num_packets, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'in'\n";
    std::terminate();
  }
  if((out = malloc_host<UDP_t>(num_packets, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'out'\n";
    std::terminate();
  }

  // create random input data
  for (size_t i = 0; i < num_packets; i++) {
    for (size_t j = 0; j < elements_per_udp_packet; j++) {
      ((T*)in[i].data)[j] = in_data[i*elements_per_udp_packet + j];
    }
  }

  // Declare the host producer and consumer which will produce and consume
  // 'fake' UDP data
  using MyProducer = HostProducer<class UDPProducerClass,
                                  class UDPProducerPipeClass,
                                  UDP_t>;
  using MyConsumer = HostConsumer<class UDPConsumerClass,
                                  class UDPConsumerPipeClass,
                                  UDP_t>;
  MyProducer producer(q, in);
  MyConsumer consumer(q, out);

  // start the producer and consumer
  event producer_event = producer.Start(num_packets);
  event consumer_event = consumer.Start(num_packets);

  // start the UDP reader/writer
  event udp_reader_event = SubmitUDPReaderKernel<class UDPReader,
                                                T,
                                                elements_per_udp_packet,
                                                udp_bytes,
                                                typename MyProducer::Pipe,
                                                InputPipe>(q, num_packets);

  event udp_writer_event = SubmitUDPWriterKernel<class UDPWriter,
                                                T,
                                                elements_per_udp_packet,
                                                udp_bytes,
                                                OutputPipe,
                                                typename MyConsumer::Pipe>(q, num_packets);


  // start the main processing kernel (pass through)
  event kernel_event = q.submit([&](handler& h) {
    h.single_task<class Wire>([=]() {
      for (size_t i = 0; i < num_packets*elements_per_udp_packet; i++) {
        T data = InputPipe::read();
        OutputPipe::write(data);
      }
    });
  });

  // wait on everything to finish
  producer_event.wait();
  consumer_event.wait();
  udp_reader_event.wait();
  udp_writer_event.wait();
  kernel_event.wait();

  // validate the output
  bool passed = true;
  for (size_t i = 0; i < num_packets; i++) {
    for (size_t j = 0; j < elements_per_udp_packet; j++) {
      // the input and output data
      T input_data = ((T*)in[i].data)[j];
      T output_data = ((T*)out[i].data)[j];

      if (!AlmostEqual(output_data, input_data)) {
        std::cerr << "ERROR: output is invalid at index " << i << ": ";
        PrintComplex(output_data);
        std::cerr << " != ";
        PrintComplex(input_data);
        std::cerr << " (Result != Expected)\n";
        passed = false;
        break;
      }
    }
  }

  // validate SOF and EOF
  if (!out[0].isSOF()) {
    std::cerr << "First packet did not have SOF bit set!\n";
    passed = false;
  }
  if (!out[num_packets-1].isEOF()) {
    std::cerr << "Last packet did not have EOF bit set!\n";
    passed = false;
  }

  std::cout << "UDP Read/Write Test: ";
  if (passed) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
  }

  // free USM memory
  sycl::free(in, q);
  sycl::free(out, q);
}
