#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

// Allow design parameters to be defined on the command line

// check is USM host allocations are enabled
#ifdef USM_HOST_ALLOCATIONS
constexpr bool kUseUSMHostAllocation = true;
#else
constexpr bool kUseUSMHostAllocation = false;
#endif

// Large array is 64 sensors.  This is not the default because the Quartus
// compile time at this size is very long (> 24 hours)
#ifdef LARGE_SENSOR_ARRAY
#define NUM_SENSORS 64
#endif

// number of sensors (= number of columns for QRD)
// Default is the 'small' size of 16 sensors
#ifndef NUM_SENSORS
#define NUM_SENSORS 16
#endif

// RMB factor (number of rows for QRD = NUM_SENSORS * RMB_FACTOR)
// Typically 3 or 5, larger number gives better SNR
#ifndef RMB_FACTOR
#define RMB_FACTOR 3
#endif

// use the SMALL_INPUT setting if the input pipe is unable to keep up and
// you want to reduce the bandwidth requirements without reducing the number
// of training matrices/s that can be processed
#ifdef SMALL_INPUT
#define NUM_INPUT_VECTORS NUM_SENSORS
#endif

// Number of input vectors to process with each training matrix
// Default to use the same size as the training matrix
#ifndef NUM_INPUT_VECTORS
#define NUM_INPUT_VECTORS (NUM_SENSORS * RMB_FACTOR)
#endif

// Number of steering vectors
#ifndef NUM_STEER
#define NUM_STEER 25
#endif

// Number of complex numbers received on each read from the streaming input
// pipe (= width of pipe in bytes / 8 )
// Must be a power of 2.  A value less than 4 may cause the input pipe to be
// the limiting factor on throughput
#ifndef STREAMING_PIPE_WIDTH
#define STREAMING_PIPE_WIDTH 4
#endif

// Degree of folding for substitution kernels
#ifndef SUBSTITUTION_FOLDING_FACTOR
#define SUBSTITUTION_FOLDING_FACTOR 1  // default to full unroll
#endif

// Degree of folding for beamformer kernel
#ifndef BEAMFORMING_FOLDING_FACTOR
#define BEAMFORMING_FOLDING_FACTOR 1  // default to full unroll
#endif

// minimum iterations for QRD
#ifndef QRD_MIN_ITERATIONS
#define QRD_MIN_ITERATIONS 80
#endif

// constants
constexpr int kNumSensorInputs = NUM_SENSORS;
constexpr int kRMBFactor = RMB_FACTOR;
constexpr int kTrainingMatrixNumRows = NUM_SENSORS * RMB_FACTOR;
constexpr int kNumInputVectors = NUM_INPUT_VECTORS;
constexpr int kNumSteer = NUM_STEER;
constexpr int kNumComplexPerXrxPipe = STREAMING_PIPE_WIDTH;
constexpr int kSubstitutionUnrollFactor =
    NUM_SENSORS / SUBSTITUTION_FOLDING_FACTOR;
constexpr int kBeamformingUnrollFactor =
    NUM_SENSORS / BEAMFORMING_FOLDING_FACTOR;
constexpr int kQRDMinIterations = QRD_MIN_ITERATIONS;

#endif  // __CONSTANTS_HPP__
