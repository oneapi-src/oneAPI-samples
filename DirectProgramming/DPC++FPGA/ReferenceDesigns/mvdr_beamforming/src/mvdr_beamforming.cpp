// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

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

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "Constants.hpp"
#include "FakeIOPipes.hpp"
#include "MVDR.hpp"

// We will have the producer and consumer use USM host allocations
// if they are enabled, otherwise they use device allocations
template <typename Id, typename T, size_t min_capacity = 0>
using MyProducer = Producer<Id, T, kUseUSMHostAllocation, min_capacity>;
template <typename Id, typename T, size_t min_capacity = 0>
using MyConsumer = Consumer<Id, T, kUseUSMHostAllocation, min_capacity>;

using namespace sycl;
using namespace std::chrono_literals;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
// utility functions
void PrintComplex(const ComplexType &c) {
  std::cout << "(" << c.real() << ", " << c.imag() << ")";
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
class XrxTrainingProducerID;
class XrxDataProducerID;
class SinThetaProducerID;
class UpdateSinThetaProducerID;
class DataOutConsumerID;

constexpr size_t kTrainingDataSize =
    kNumSensorInputs * kTrainingMatrixNumRows / kNumComplexPerXrxPipe;
constexpr size_t kInputDataSize =
    kNumSensorInputs * kNumInputVectors / kNumComplexPerXrxPipe;
constexpr size_t kDataOutSize = kNumInputVectors * kNumSteer;

using XrxPipeType = NTuple<ComplexType, kNumComplexPerXrxPipe>;

// File I/O
bool ReadInputData(std::string in_dir, ComplexType *training_data,
                   ComplexType *data_in, int num_matrix_copies);
bool WriteOutputData(std::string out_dir, ComplexType *data_out);
bool CheckOutputData(std::string in_dir, ComplexType *data_out,
                     bool print_diffs);

// the main function
int main(int argc, char *argv[]) {
  std::string in_dir = "../data";
  std::string out_dir = ".";
  int num_matrix_copies = 1024;  // number of matrices for throughput test

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      std::cout << "USAGE: ./mvdr_beamforming "
                << "[--in=<string>] "
                << "[--out=<string>]\n";
      return 0;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--in=") == 0) {
        in_dir = std::string(str_after_equals.c_str());
      } else if (arg.find("--out=") == 0) {
        out_dir = std::string(str_after_equals.c_str());
      } else if (arg.find("--n=") == 0) {
        num_matrix_copies = std::stoi(str_after_equals);
      } else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  bool passed = true;
  bool one_passed;

  // start the more 'SYCL' portion of the design
  try {
    // device selector
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // queue properties to enable SYCL profiling of kernels
    auto prop_list = property_list{property::queue::enable_profiling()};

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // TODO training and data need to be sent through the same pipe, and split
    // by a kernel

    // host producer and consumers
    using XrxTrainingProducer =
        MyProducer<XrxTrainingProducerID, XrxPipeType, kTrainingDataSize * 2>;
    using XrxTrainingPipe = XrxTrainingProducer::Pipe;

    using XrxDataProducer =
        MyProducer<XrxDataProducerID, XrxPipeType, kInputDataSize * 2>;
    using XrxDataPipe = XrxDataProducer::Pipe;

    using SinThetaProducer =
        MyProducer<SinThetaProducerID, float, kNumSteer * 2>;
    using SinThetaPipe = SinThetaProducer::Pipe;

    using DataOutConsumer =
        MyConsumer<DataOutConsumerID, ComplexType, kDataOutSize * 2>;
    using DataOutPipe = DataOutConsumer::Pipe;

    // initialize the producers and consumers
    XrxTrainingProducer::Init(q, kTrainingDataSize * num_matrix_copies);
    XrxDataProducer::Init(q, kInputDataSize * num_matrix_copies);
    SinThetaProducer::Init(q, kNumSteer);
    DataOutConsumer::Init(q, kDataOutSize * num_matrix_copies);

    // clear the output data
    std::fill_n(DataOutConsumer::Data(), kDataOutSize * num_matrix_copies, 0.0);

    // read the input data
    passed &= ReadInputData(in_dir, (ComplexType *)XrxTrainingProducer::Data(),
                            (ComplexType *)XrxDataProducer::Data(),
                            num_matrix_copies);

    // calcluate the sin(theta) values for each steering vector
    constexpr float degree_unit = 120.0f / (kNumSteer - 1);
    for (int i = 0; i < kNumSteer; i++) {
      float degree = -60.0f + i * degree_unit;
      SinThetaProducer::Data()[i] = sin(degree / 180.0f * M_PI);
    }

    std::cout << "Calculated sin(theta) values" << std::endl;

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
        XrxTrainingPipe,            // Sensor data sent to the training kernels
        XrxDataPipe,                // Sensor data sent to beamforming kernel
        SinThetaPipe,               // sin(theta) input for steering vectors
        DataOutPipe                 // output from MVDR
        >(q, kNumInputVectors);

    std::cout << "Launched MVDR kernels" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // Basic test
    // Send all data sources, check output
    // Also write output to a file
    ////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl
              << "*** Basic single matrix and steering vectors test ***"
              << std::endl;

    // start the consumer
    std::cout << "Launching consumer kernel" << std::endl;
    event data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);

    // start the producers
    std::cout << "Launching producer kernels" << std::endl;
    event sin_theta_producer_event = SinThetaProducer::Start(q, kNumSteer);
    event xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    event xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);

    // wait for producer kernels to finish
    sin_theta_producer_event.wait();
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    std::cout << "Producer kernels finished" << std::endl;

    // wait for consumer kernels to finish
    data_out_consumer_event.wait();
    std::cout << "Consumer kernels finished" << std::endl;

    // write the output to a file
    passed &= WriteOutputData(out_dir, DataOutConsumer::Data());

    // check the output against expected result
    one_passed = CheckOutputData(in_dir, DataOutConsumer::Data(), true);
    if (one_passed) {
      std::cout << "Output data check succeeded" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }
    passed &= one_passed;

    ////////////////////////////////////////////////////////////////////////////
    // Re-send Xrx and training data test
    // Don't update training data or sin(theta), so should get same result
    ////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl << "*** Re-send single matrix test ***" << std::endl;

    // start the producers
    std::cout << "Re-sending Xrx and training data" << std::endl;
    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    std::cout << "Producer kernels finished" << std::endl;
    data_out_consumer_event.wait();
    std::cout << "Consumer kernels finished" << std::endl;

    one_passed = CheckOutputData(in_dir, DataOutConsumer::Data(), true);
    if (one_passed) {
      std::cout << "Output data check succeeded" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }
    passed &= one_passed;

    ////////////////////////////////////////////////////////////////////////////
    // Modify weight vectors, re-send Xrx and training data, expect a mismatch
    ////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl
              << "*** Modify weight vectors test (expect data mismatch) ***"
              << std::endl;
    std::cout << "Modifying and sending sin(theta) values" << std::endl;
    float *sin_theta_ptr = SinThetaProducer::Data();

    float sin_theta_zero_save = sin_theta_ptr[0];
    sin_theta_ptr[0] = sin_theta_ptr[1];
    sin_theta_producer_event = SinThetaProducer::Start(q, kNumSteer);
    sin_theta_producer_event.wait();

    std::cout << "Re-sending Xrx and training data three times" << std::endl;
    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    one_passed = !CheckOutputData(in_dir, DataOutConsumer::Data(), false);
    if (one_passed) {
      std::cout << "Output data mismatched as expected" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }
    passed &= one_passed;

    ////////////////////////////////////////////////////////////////////////////
    // Restore weight vectors, re-send Xrx and training data, expect a match
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Restoring original sin(theta)[0] value" << std::endl;
    sin_theta_ptr[0] = sin_theta_zero_save;
    sin_theta_producer_event = SinThetaProducer::Start(q, kNumSteer);
    sin_theta_producer_event.wait();

    std::cout << "Re-sending Xrx and training data three times" << std::endl;
    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    data_out_consumer_event = DataOutConsumer::Start(q, kDataOutSize);
    xrx_data_producer_event = XrxDataProducer::Start(q, kInputDataSize);
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize);
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    one_passed = CheckOutputData(in_dir, DataOutConsumer::Data(), true);
    if (one_passed) {
      std::cout << "Output data check succeeded" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }
    passed &= one_passed;

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

    // start the consumer
    data_out_consumer_event =
        DataOutConsumer::Start(q, kDataOutSize * num_matrix_copies);

    auto start_time = high_resolution_clock::now();

    // start the producers
    xrx_training_producer_event =
        XrxTrainingProducer::Start(q, kTrainingDataSize * num_matrix_copies);
    xrx_data_producer_event =
        XrxDataProducer::Start(q, kInputDataSize * num_matrix_copies);

    // wait for producer and consumer kernels to finish
    xrx_training_producer_event.wait();
    xrx_data_producer_event.wait();
    data_out_consumer_event.wait();

    auto end_time = high_resolution_clock::now();

    // compute latency and throughput
    duration<double, std::milli> process_time = end_time - start_time;
    double latency_s = process_time.count() / 1000.0;
    double throughput = (double)num_matrix_copies / latency_s;

    std::cout << "Throughput: " << throughput << " matrices/second"
              << std::endl;

    // check one instance of output data
    one_passed = CheckOutputData(in_dir, DataOutConsumer::Data(), true);
    if (one_passed) {
      std::cout << "Output data check succeeded" << std::endl;
    } else {
      std::cout << "Output data check failed" << std::endl;
    }
    passed &= one_passed;

    ////////////////////////////////////////////////////////////////////////////
    // Post-test cleanup
    ////////////////////////////////////////////////////////////////////////////

    // free the host USM memory
    XrxTrainingProducer::Destroy(q);
    XrxDataProducer::Destroy(q);
    SinThetaProducer::Destroy(q);
    DataOutConsumer::Destroy(q);

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

bool ReadInputData(std::string in_dir, ComplexType *training_data,
                   ComplexType *data_in, int num_matrix_copies) {
  // file paths relative the the base directory
  std::string training_real_path = in_dir + "/" + "A_real.txt";
  std::string training_imag_path = in_dir + "/" + "A_imag.txt";
  std::string x_real_path = in_dir + "/" + "X_real.txt";
  std::string x_imag_path = in_dir + "/" + "X_imag.txt";

  ComplexBaseType real, imag;

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

  // load the first matrix from the input file
  // TODO temporarily, we do the transpose of this matrix in host code, until
  // the transpose kernel is fully implemented data comes in row order, we need
  // it in column order
  // Note this is not a 'full' transpose, we just transpose
  // kNumComplexPerXrxPipe rows at a time
  for (size_t row = 0; row < kTrainingMatrixNumRows; row++) {
    for (size_t col = 0; col < kNumSensorInputs; col++) {
      a_real_is >> real;
      a_imag_is >> imag;
      size_t data_index = (row / kNumComplexPerXrxPipe) * kNumSensorInputs *
                              kNumComplexPerXrxPipe +
                          col * kNumComplexPerXrxPipe +
                          row % kNumComplexPerXrxPipe;
      training_data[data_index].set_r(real);
      training_data[data_index].set_i(imag);
    }
  }

  a_real_is.close();
  a_imag_is.close();

  // copy the first training matrix num_matrix_copies times
  for (size_t matrix_num = 1; matrix_num < num_matrix_copies; matrix_num++) {
    for (size_t i = 0; i < kNumSensorInputs * kTrainingMatrixNumRows; i++) {
      size_t data_copy_index =
          i + (matrix_num * kNumSensorInputs * kTrainingMatrixNumRows);
      training_data[data_copy_index] = training_data[i];
    }
  }

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

  // load the first matrix
  for (size_t i = 0; i < kInputDataSize * kNumComplexPerXrxPipe; i++) {
    x_real_is >> real;
    x_imag_is >> imag;
    data_in[i].set_r(real);
    data_in[i].set_i(imag);
  }

  x_real_is.close();
  x_imag_is.close();

  // copy the first data matrix num_matrix_copies times
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
                     bool print_diffs) {
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

  for (size_t i = 0; i < kNumInputVectors; i++) {
    for (size_t j = 0; j < kNumSteer; j++) {
      ComplexType expected;
      ComplexBaseType data_in;
      exp_real_is >> data_in;
      expected.set_r(data_in);
      exp_imag_is >> data_in;
      expected.set_i(data_in);
      if (!AlmostEqual(expected, data_out[i * kNumSteer + j])) {
        if (print_diffs) {
          std::cout << "Error at input vector " << i << " steering vector " << j
                    << ". Expected: " << expected.real() << " + "
                    << expected.imag() << "i, "
                    << " Actual: " << data_out[i * kNumSteer + j].real()
                    << " + " << data_out[i * kNumSteer + j].imag() << "i"
                    << std::endl;
        }
        match = false;
      }
    }
  }

  exp_real_is.close();
  exp_imag_is.close();

  return match;
}

bool WriteOutputData(std::string out_dir, ComplexType *data_out) {
  // file paths relative the the base directory
  std::string out_real_path = out_dir + "/" + "out_real.txt";
  std::string out_imag_path = out_dir + "/" + "out_imag.txt";

  ComplexBaseType real, imag;

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
