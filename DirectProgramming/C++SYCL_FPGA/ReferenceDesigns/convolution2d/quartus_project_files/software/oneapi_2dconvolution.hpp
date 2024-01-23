#pragma once

// macros for BSP
#include <system.h>
// macros for oneAPI kernel
#include "../../whitepau.convolution2d/build/convolution.report.prj/include/register_map_offsets.hpp"
// IORD and IOWR
#include <io.h>

// integer types
#include <stdint.h>

#define KERNEL_START                        \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_START_REG)
#define KERNEL_STOP                         \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_STOPCSR_PIPE_CHANNEL_DATA_REG)
#define KERNEL_STOP_VALID                   \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_STOPCSR_PIPE_CHANNEL_VALID_REG)

#define KERNEL_FINISH_COUNTER               \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_FINISHCOUNTER_REG)

#define KERNEL_STATUS_REGISTER              \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
  ZTS16ID_CONVOLUTION2D_REGISTER_MAP_STATUS_REG)

#define KERNEL_INFO                         \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
  ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_INFOCSR_PIPE_CHANNEL_DATA_REG)

#define KERNEL_INFO_VALID                   \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
  ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_INFOCSR_PIPE_CHANNEL_VALID_REG)

#define KERNEL_BYPASS                       \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_BYPASSCSR_PIPE_CHANNEL_DATA_REG)
#define KERNEL_BYPASS_VALID                 \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ACL_C_ID_BYPASSCSR_PIPE_CHANNEL_VALID_REG)

#define ARG_CONVKERNEL                      \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_CONVKERNEL_0_REG)

#define ARG_ROWS                            \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_ROWS_REG)

// bug in oneAPI: kernel arguments do not have accurate byte addresses.
#define ARG_COLS                            \
  (M_VVP_PIPELINE_ONEAPI_CONVOLUTION_BASE + \
   ZTS16ID_CONVOLUTION2D_REGISTER_MAP_ARG_ARG_COLS_REG + 4)


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

void start() {
  IOWR_32DIRECT(KERNEL_STOP, 0, 0x0);
  IOWR_32DIRECT(KERNEL_STOP_VALID, 0, 0x1);
  IOWR_32DIRECT(KERNEL_START, 0, 0x1);
}

void stop() {
  IOWR_32DIRECT(KERNEL_STOP, 0, 0x1);
  IOWR_32DIRECT(KERNEL_STOP_VALID, 0, 0x1);
}

int getStopCount() {
  return IORD_32DIRECT(KERNEL_FINISH_COUNTER, 0);
}

// set convolution kernel

void set_convolution_coeffs(size_t len, uint8_t *coeffs)
{
  printf("Setting coefficients:\n\t");
  // copy byte-wise to register
  for(size_t i = 0; i < len; i++) {
    printf("0x%x ", coeffs[i]);
    IOWR_8DIRECT(ARG_CONVKERNEL + i, 0, coeffs[i]);
  }
  printf("\n");
}

void init_coeffs_identity3x3() {
  float coeffs[9] = K_COEFFS_IDENTITY;

  // re-cast to int to be compatible with the macro
  uint8_t *coeffs_bytes = reinterpret_cast<uint8_t *>(coeffs);
  set_convolution_coeffs(9 * sizeof(float), coeffs_bytes);
}


void init_coeffs_sobel_vertical() {
  float coeffs[9] = K_COEFFS_SOBEL_VERTICAL;

  // re-cast to int to be compatible with the macro
  uint8_t *coeffs_bytes = reinterpret_cast<uint8_t *>(coeffs);
  set_convolution_coeffs(9 * sizeof(float), coeffs_bytes);
}

uint64_t pollStatusRegister() {
  uint64_t status_register_data;
  uint32_t *sr_low = reinterpret_cast<uint32_t *>(&status_register_data);
  uint32_t *sr_high = &sr_low[1];
  *sr_low = IORD_32DIRECT(KERNEL_STATUS_REGISTER, 0);
  *sr_high = IORD_32DIRECT(KERNEL_STATUS_REGISTER, 4);

  return status_register_data;
}

uint32_t pollFinishCounter() {
  return IORD_32DIRECT(KERNEL_FINISH_COUNTER, 0);
}

uint32_t getKernelVersion() {
  return IORD_32DIRECT(KERNEL_INFO, 0);
}

void setBypass(bool bypass)
{
  IOWR_32DIRECT(KERNEL_BYPASS, 0, bypass);
  IOWR_32DIRECT(KERNEL_BYPASS_VALID, 0, 0x1);
}

void setRows(int rows) {
  IOWR_32DIRECT(ARG_ROWS, 0, rows);
}

void setCols(int cols) {
  IOWR_32DIRECT(ARG_COLS, 0, cols);
}
}  // namespace oneapiconvolution