#pragma once

// macros for BSP
#include <system.h>
// macros for oneAPI kernel
#include "<path to convolution2d sample>/build/conv.report.prj/include/register_map_offsets.h"
// IORD and IOWR
#include <io.h>

// integer types
#include <stdint.h>

// Modify this base to refer to the macro declared in system.h
#define CONVOLUTION_BASE CONVOLUTION_ONEAPI_BASE

#define KERNEL_START \
  (CONVOLUTION_BASE + ID_CONVOLUTION2D_REGISTER_MAP_START_REG)

#define KERNEL_STOP   \
  (CONVOLUTION_BASE + \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_STOPCSR_PIPE_CHANNEL_DATA_REG)
#define KERNEL_STOP_VALID \
  (CONVOLUTION_BASE +     \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_STOPCSR_PIPE_CHANNEL_VALID_REG)

#define KERNEL_STATUS_REGISTER \
  (CONVOLUTION_BASE + ID_CONVOLUTION2D_REGISTER_MAP_STATUS_REG)

#define KERNEL_VERSION \
  (CONVOLUTION_BASE +  \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_VERSIONCSR_PIPE_CHANNEL_DATA_REG)
#define KERNEL_VERSION_VALID \
  (CONVOLUTION_BASE +        \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_VERSIONCSR_PIPE_CHANNEL_VALID_REG)

#define KERNEL_BYPASS \
  (CONVOLUTION_BASE + \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_BYPASSCSR_PIPE_CHANNEL_DATA_REG)
#define KERNEL_BYPASS_VALID \
  (CONVOLUTION_BASE +       \
   ID_CONVOLUTION2D_REGISTER_MAP_ACL_C_ID_BYPASSCSR_PIPE_CHANNEL_VALID_REG)

#define ARG_COEFFS \
  (CONVOLUTION_BASE + ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_0_REG)

#define ARG_COEFFS_SIZE ( \
  ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_0_SIZE + \
  ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_1_SIZE + \
  ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_2_SIZE + \
  ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_3_SIZE + \
  ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COEFFS_4_SIZE   \
)

#define ARG_ROWS \
  (CONVOLUTION_BASE + ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_ROWS_REG)

#define ARG_COLS \
  (CONVOLUTION_BASE + ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COLS_REG)

namespace oneapiconvolution {

// clang-format off
#define K_COEFFS_SOBEL_VERTICAL        \
  {                                    \
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,   \
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,   \
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f    \
  }

#define K_COEFFS_IDENTITY   \
  {                         \
    0.0f, 0.0f, 0.0f,       \
    0.0f, 1.0f, 0.0f,       \
    0.0f, 0.0f, 0.0f        \
  }

////////////////////////////////////////////////////////////////////////////////
// Accessor/mutator functions for SYCL HLS IP
//
// The following functions use de-referenced pointers to access the CSR for the
// convolution2d IP. This is fine so long as the CSR is mapped within the
// peripheral region of the Nios® V softcore processor. Double-check your
// Platform Designer configuration to verify this.
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// read status register
/////////////////////////

uint64_t poll_status_register() {
  volatile uint64_t *kernel_status_ptr = (volatile uint64_t *)KERNEL_STATUS_REGISTER;
  return *kernel_status_ptr;
}

////////////////////////////////
// read from an output CSR pipe
////////////////////////////////

// The kernel will only write its version to its CSR after being started, so
// make sure that you have called `start()` at least once before polling the
// version.
bool get_kernel_version(int32_t &kernel_version) {
  // the type of the version is indicated in the SYCL HLS kernel source code.
  volatile int32_t *version_ptr = (volatile int32_t *)KERNEL_VERSION;
  kernel_version = *version_ptr;
  return true;
}

////////////////////////
// set kernel arguments
////////////////////////

void set_rows(int32_t rows) {
  volatile int32_t *rows_ptr = (volatile int32_t *)ARG_ROWS;
  *rows_ptr = rows;
}

void set_cols(int32_t cols) {
  volatile int32_t *cols_ptr = (volatile int32_t *)ARG_COLS;
  *cols_ptr = cols;
}

// Return 'false' if the coefficients array is too large. This could cause buffer
// overflow.
bool set_convolution_coeffs(size_t len, uint8_t *coeffs)
{
  if (len <= ARG_COEFFS_SIZE){
    volatile uint8_t *arg_coeffs = (volatile uint8_t *)ARG_COEFFS;

    // explicit copy loop since memcpy does not support volatile
    for(size_t i = 0; i < len; i++) {
      arg_coeffs[i] = coeffs[i];
    }
    return true;
  } else {
    return false;
  }
}

// Start the kernel if it is not already running. Return 'true' if the kernel was
// not already running. 
bool start() {
  uint64_t status_register = poll_status_register();
  bool running = status_register & KERNEL_REGISTER_MAP_RUNNING_MASK;

  volatile uint64_t *kernel_start_ptr = (volatile uint64_t *)KERNEL_START;
  if (!running) {
    kernel_start_ptr[0] = 0x1;
    return true;
  } else 
  {
    return false;
  }  
}

//////////////////////////////////////////////////////////////////////////////
// The next two functions modify a CSR that is connected to a pipe in the
// kernel. Therefore, you must make sure to set the 'valid' register to 'true' 
// so the kernel knows there is new data.
//////////////////////////////////////////////////////////////////////////////

// Return 'true' if the kernel was running. It is not appropriate to try to stop
// a kernel that isn't already running.
bool stop() {
  uint64_t status_register = poll_status_register();
  bool running = status_register & KERNEL_REGISTER_MAP_RUNNING_MASK;

  if (running) {
    volatile bool *stop_valid = (volatile bool *)KERNEL_STOP_VALID;
    volatile bool *stop       = (volatile bool *)KERNEL_STOP;

    *stop       = true;
    *stop_valid = true;
  }
  return running;
}

// Return 'true' if the kernel was ready to consume the data
bool set_bypass(bool bypass_new)
{
  volatile bool *bypass_valid = (volatile bool *)KERNEL_BYPASS_VALID;
  volatile bool *bypass       = (volatile bool *)KERNEL_BYPASS;

  *bypass       = bypass_new;
  *bypass_valid = true;
  return true;
}

/////////////////////////////////////////////////////////////////////////////
// The following convenience functions let you load pre-computed coefficient
// sets to the IP.
/////////////////////////////////////////////////////////////////////////////

bool init_coeffs_identity_3x3() {
  float coeffs[9] = K_COEFFS_IDENTITY;

  // re-cast to int to be compatible with the macro
  uint8_t *coeffs_bytes = reinterpret_cast<uint8_t *>(coeffs);
  return set_convolution_coeffs(9 * sizeof(float), coeffs_bytes);
}


bool init_coeffs_sobel_vertical() {
  float coeffs[9] = K_COEFFS_SOBEL_VERTICAL;

  // re-cast to int to be compatible with the macro
  uint8_t *coeffs_bytes = reinterpret_cast<uint8_t *>(coeffs);
  return set_convolution_coeffs(9 * sizeof(float), coeffs_bytes);
}
}  // namespace oneapiconvolution