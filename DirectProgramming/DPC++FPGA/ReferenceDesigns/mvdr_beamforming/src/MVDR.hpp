#ifndef __MVDR_HPP__
#define __MVDR_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <array>

// utility classes
#include "pipe_array.hpp"
#include "NullPipe.hpp"
#include "PipeAggregator.hpp"

#include "mvdr_complex.hpp"

// MVDR processing kernels
#include "Transpose.hpp"
#include "StreamingQRD.hpp"
#include "SteeringVectorGenerator.hpp"
#include "ForwardSubstitution.hpp"
#include "BackwardSubstitution.hpp"
#include "CalcWeights.hpp"
#include "Beamformer.hpp"

using namespace sycl;

// Names of kernels launched by SubmitMVDRKernels.
// This enum should be used to access elements in the returned vector of events.
enum class MVDRKernelNames {
  transpose,
  streaming_qrd,
  steering_vector_generator,
  forward_substitution,
  backward_substitution,
  calc_weights,
  beamformer,

  // count must come last
  count
};

using MVDREventArray = 
  std::array<event, static_cast<int>(MVDRKernelNames::count)>;


// forward declare the names of all kernels to prevent name mangling
template <size_t k_instance_num> class Transpose;
template <size_t k_instance_num> class StreamingQRD;
template <size_t k_instance_num> class SteeringVectorGenerator;
template <size_t k_instance_num> class ForwardSubstitution;
template <size_t k_instance_num> class BackwardSubstitution;
template <size_t k_instance_num> class CalcWeights;
template <size_t k_instance_num> class Beamformer;

// forward declare the names of all pipes to prevent name mangling
template <size_t k_instance_num> class SteeringVectorsPipeID;
template <size_t k_instance_num> class UpdateSteeringVectorsPipeID;
template <size_t k_instance_num> class ForwardSteeringVectorsPipeID;
template <size_t k_instance_num> class QMatrixPipeID;
template <size_t k_instance_num> class RMatrixPipesID;
template <size_t k_instance_num> class RDiagRecipVectorPipesID;
template <size_t k_instance_num> class ForwardSubstitutionResultPipeID;
template <size_t k_instance_num> class YVectorsPipeID;
template <size_t k_instance_num> class WeightVectorsPipeID;
template <size_t k_instance_num> class TrainingArrayPipeID;


// SubmitMVDRKernels
// Launch all the kernels to perform MVDR processing.
// Return a vector of events, one for each kernel
template <
  size_t    k_num_sensor_inputs,      // number of sensor array inputs
  size_t    k_rmb_factor,             // Reed-Mallett-Brennan rule
                                      // Number of 'rows' of sensor data used
                                      // by the QRD is k_num_sensor_inputs *
                                      // k_rmb_factor (generally 2-5)
  size_t    k_num_steering_vectors,   // number of steering vectors to apply to
                                      // each input sample
  size_t    k_subst_unroll_factor,    // unroll factor used by the forward and
                                      // backward substitution kernels
  size_t    k_beam_unroll_factor,     // unroll factor used by beamformer
  size_t    k_qrd_min_iterations,     // minimum 'inner loop' iterations for the
                                      // QRD kernel, this number can be tuned
                                      // for best throughput
  size_t  k_num_complex_per_xrx_read, // Number of complex numbers (contained
                                      // in NTuple) per read from the
                                      // Xrx input pipes

  typename  XrxTrainingInPipe,        // Sensor data to be sent to the MVDR
                                      // training kernels.  A matrix of
                                      // k_num_sensor_inputs by k_rmb_factor
                                      // complex floats is used to generate a
                                      // new set of weight vectors.
                                      // Receive a NTuple containing 
                                      // k_num_complex_per_xrx_read complex
                                      // floats per read
  typename  XrxDataInPipe,            // Sensor data to be sent to the MVDR
                                      // beamformer kernel.  Each vector of
                                      // k_num_sensor_inputs complex floats
                                      // will be multiplied by all weight
                                      // vectors.
                                      // Receive a NTuple containing 
                                      // k_num_complex_per_xrx_read complex
                                      // floats per read
  typename  SinThetaInPipe,           // sin(theta) input for generating
                                      // steering vectors. Updated by another
                                      // kernel with updates from the host.
                                      // Accept one float per read.
  typename  DataOutPipe,              // For each Xrx input data vector, send
                                      // an output for each of the Weight
                                      // vectors.
                                      // Send one complex float per write.
  size_t    k_instance_num = 0        // To allow more than one MVDR instance
                                      // in a system, provide a unique
                                      // instance_num to each.
>
MVDREventArray SubmitMVDRKernels( 
  queue&  q,
  short   num_xrx_per_weights         // Number of xrx vectors to process with
                                      // each set of Weight vectors.
) {
  constexpr short kNumTrainingRows = k_num_sensor_inputs * k_rmb_factor;

  // Template parameter checking
  // Most template parameters will be checked in individual kernels
  static_assert( k_num_sensor_inputs > 0,
    "k_num_sensor_inputs must be greater than 0" );
  static_assert( k_rmb_factor > 0,
    "k_rmb_factor must be greater than 0" );
  static_assert(std::numeric_limits<short>::max() > kNumTrainingRows,
    "k_num_sensor_inputs * k_rmb_factor must fit in a short" );
  

  // Steering vector generator and related update pipe
  // Connect SteeringVectorGenerator to ForwardSubstitution
  constexpr short kSteeringVectorsPipeMinDepth = 
    k_num_steering_vectors * k_num_sensor_inputs * 2;
  using SteeringVectorsPipe = sycl::pipe< SteeringVectorsPipeID<k_instance_num>, 
                                          ComplexType, 
                                          kSteeringVectorsPipeMinDepth >;
  using UpdateSteeringVectorsPipe = sycl::pipe< 
    UpdateSteeringVectorsPipeID<k_instance_num>, bool, 1 >;

  // Pipe for forwarding steering vectors used by ForwardSubstitution to
  // CalcWeights
  using ForwardSteeringVectorsPipe = sycl::pipe<
    ForwardSteeringVectorsPipeID<k_instance_num>, 
    ComplexType, kSteeringVectorsPipeMinDepth >;

  // R matrix and R matrix reciprocal diagonal entries pipes
  // Connect StreamingQRD to ForwardSubstitution and BackwardSubstitution
  // Need two copies of each so create 1D arrays of Pipes
  // Min depth ensures we can hold 2 full R matricies in the pipe, to make sure
  // pipe feeding BackwardSubstitution won't overflow while waiting for result
  // from ForwardSubstitution.
  constexpr short kRMatrixPipeMinDepth = 
    (( k_num_sensor_inputs * (k_num_sensor_inputs + 1) ) / 2) * 2;
  using RMatrixPipes = PipeArray< 
    RMatrixPipesID<k_instance_num>, ComplexType, kRMatrixPipeMinDepth, 2 >;
  constexpr short kRDiagRecipVectorPipeMinDepth = k_num_sensor_inputs * 2;
  using RDiagRecipVectorPipes = PipeArray< 
    RDiagRecipVectorPipesID<k_instance_num>, 
    float, kRDiagRecipVectorPipeMinDepth, 2 >;
  using RMatrixFSPipe = typename RMatrixPipes::template PipeAt<0>;
  using RMatrixBSPipe = typename RMatrixPipes::template PipeAt<1>;
  using RDiagRecipVectorFSPipe =
    typename RDiagRecipVectorPipes::template PipeAt<0>;
  using RDiagRecipVectorBSPipe =
    typename RDiagRecipVectorPipes::template PipeAt<1>;

  // Forward substitution result pipe
  // Connect ForwardSubstitution to BackwardSubstitution
  using ForwardSubstitutionResultPipe = sycl::pipe< 
    ForwardSubstitutionResultPipeID<k_instance_num>,
    ComplexType, k_num_sensor_inputs >;

  // Y vectors pipe
  // Y = (inverse(R x Rtranspose) ) * (complex_conjugate(C)) , where 
  // R is the R matrix from QRD, and C is the steering vector
  // Connect BackwardSubstitution to CalcWeights
  using YVectorsPipe = sycl::pipe< 
    YVectorsPipeID<k_instance_num>, ComplexType, k_num_sensor_inputs >;

  // Weight vectors pipe
  // Connect CalcWeights to Beamformer
  constexpr short kWeightVectorsPipeMinDepth = 
    k_num_steering_vectors * k_num_sensor_inputs * 2;
  using WeightVectorsPipe = sycl::pipe< WeightVectorsPipeID<k_instance_num>,
    ComplexType, kWeightVectorsPipeMinDepth >;

  // Q matrix pipe
  // Q matrix not used in MVDR design, so this is a 'null' pipe
  using QMatrixColumn = NTuple< ComplexType, k_num_sensor_inputs >;
  using QMatrixPipe = NullPipe< QMatrixPipeID<k_instance_num>, QMatrixColumn >;

  // transposed training data pipe
  constexpr short kTrainingArrayPipeMinDepth = k_num_sensor_inputs * 
    k_num_sensor_inputs * k_rmb_factor / k_num_complex_per_xrx_read;
  using XrxPipeType = NTuple< ComplexType, k_num_complex_per_xrx_read >;
  using TrainingArrayPipe = sycl::pipe< TrainingArrayPipeID<k_instance_num>, 
                                        XrxPipeType, 
                                        kTrainingArrayPipeMinDepth >;

  // array of events to return
  // use MVDRKernelNames enum as indicies into the array
  MVDREventArray events;

  events[static_cast<int>(MVDRKernelNames::steering_vector_generator)] = 
    SubmitSteeringVectorGeneratorKernel<
      SteeringVectorGenerator<k_instance_num>,  // Name to use for the Kernel
      k_num_steering_vectors,         // number of steering vectors
      k_num_sensor_inputs,            // number of elements in each vector
      SinThetaInPipe,                 // sin(theta) input
      SteeringVectorsPipe,            // generated steering vectors
      UpdateSteeringVectorsPipe       // load new steering vectors
    >( q );

  events[static_cast<int>(MVDRKernelNames::transpose)] =
    SubmitTransposeKernel<
      Transpose<k_instance_num>,  // Name to use for the Kernel
      ComplexType,                // type of element to transpose
      k_num_sensor_inputs,        // number of columns in the input matrix
      k_num_complex_per_xrx_read, // number of elements per pipe read/write
      XrxTrainingInPipe,          // matrix input
      TrainingArrayPipe           // Output matrix
    >( q );

  events[static_cast<int>(MVDRKernelNames::streaming_qrd)] =
    SubmitStreamingQRDKernel<
      StreamingQRD<k_instance_num>,   // Name to use for the Kernel
      k_qrd_min_iterations,       // Minimum number of inner loop iterations
      kNumTrainingRows,           // Number of rows in the incoming A matrix
      k_num_sensor_inputs,        // Number of columns in the incoming A matrix
      k_num_complex_per_xrx_read, // number of elements per pipe read
      TrainingArrayPipe,          // A matrix input
      QMatrixPipe,                // Q output pipe (unused in MVDR)
      RMatrixPipes,               // R output pipe
      RDiagRecipVectorPipes       // 1 / the value of each diagonal entry of R
    >( q );

  events[static_cast<int>(MVDRKernelNames::forward_substitution)] = 
    SubmitForwardSubstitutionKernel<
      ForwardSubstitution<k_instance_num>,  // Name to use for the Kernel
      k_num_sensor_inputs,            // Number of elements in each vector
      k_subst_unroll_factor,          // inner loop unroll factor
      k_num_steering_vectors,         // Number of y vectors
      RMatrixFSPipe,                  // lower-triangular matrix L
      RDiagRecipVectorFSPipe,         // 1 / diag of L
      SteeringVectorsPipe,            // Y vectors in
      UpdateSteeringVectorsPipe,      // load new Y vectors
      ForwardSteeringVectorsPipe,     // Steering vectors used to calculate X
      ForwardSubstitutionResultPipe   // X vectors out
    >( q );

  events[static_cast<int>(MVDRKernelNames::backward_substitution)] = 
    SubmitBackwardSubstitutionKernel<
      BackwardSubstitution<k_instance_num>, // Name to use for the Kernel
      k_num_sensor_inputs,            // Number of elements in each vector
      k_subst_unroll_factor,          // inner loop unroll factor
      k_num_steering_vectors,         // Number of y vectors
      RMatrixBSPipe,                  // upper-triangular matrix U.
      RDiagRecipVectorBSPipe,         // 1 / diag of U
      ForwardSubstitutionResultPipe,  // Y vectors in
      YVectorsPipe                    // X vectors out
    >( q );

  events[static_cast<int>(MVDRKernelNames::calc_weights)] = 
    SubmitCalcWeightsKernel<
      CalcWeights<k_instance_num>,    // Name to use for the Kernel
      k_num_steering_vectors,         // number of steering vectors
      k_num_sensor_inputs,            // number of elements in each vector
      YVectorsPipe,                   // Receive the Y vectors.
      ForwardSteeringVectorsPipe,     // steering vectors
      WeightVectorsPipe               // weight vectors output
    >( q );

  events[static_cast<int>(MVDRKernelNames::beamformer)] = 
    SubmitBeamformerKernel<
      Beamformer<k_instance_num>,     // Name to use for the Kernel
      k_num_steering_vectors,         // number of weight vectors
      k_num_sensor_inputs,            // number of elements in each vector
      k_num_complex_per_xrx_read,     // complex numbers per xrx pipe read
      k_beam_unroll_factor,           // unroll factor
      XrxDataInPipe,                  // Receive the Xrx vectors
      WeightVectorsPipe,              // weight vectors input
      DataOutPipe                     // final data output
    >( q, num_xrx_per_weights );

  return events;

}   // end of SubmitMVDRKernels()

#endif  // ifndef __MVDR_HPP__
