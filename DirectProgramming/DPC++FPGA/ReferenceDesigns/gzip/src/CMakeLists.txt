# High Bandwidth variant source files
set(DEVICE_SOURCE_FILE gzipkernel.cpp)
set(DEVICE_HEADER_FILE gzipkernel.hpp)
set(HOST_SOURCE_FILE gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)

# Low Latency variant source files
set(DEVICE_SOURCE_FILE_LL gzipkernel_ll.cpp)
set(DEVICE_HEADER_FILE_LL gzipkernel_ll.hpp)
set(HOST_SOURCE_FILE_LL gzip_ll.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)

set(TARGET_NAME gzip)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(EMULATOR_TARGET_LL ${TARGET_NAME}_ll.fpga_emu)

set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_LL ${TARGET_NAME}_ll.fpga)

set(HIGH_BW_SEED "-Xsseed=8")
set(LL_SEED "-Xsseed=1")

# FPGA board selection
if (NOT DEFINED FPGA_BOARD)
    message(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for information on board selection.")
    message(STATUS "\tNote: The Low-Latency Variant of the GZIP reference design not supported on Arria(R) 10 GX.")
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
endif()

# Design specific constant values
# To increase NUM_ENGINES to greater than 2, must also statically declare more engines in gzipkernel.cpp --> SubmitGzipTasks()
if(FPGA_BOARD STREQUAL "intel_a10gx_pac:pac_a10")
    # A10 parameters
    set(NUM_ENGINES 1)
    set(NUM_REORDER "")
elseif(FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10_usm")
    # S10 parameters
    set(NUM_ENGINES 2)
    # For the High Bandwidth variant, specify 6 reordering units to improve global memory read bandwidth across 4 channels of DDR.
    # For Low Latency variant this is not necessary since only one channel of global memory is used (host memory).
    set(NUM_REORDER "-Xsnum-reorder=6")
    set(NUM_REORDER_LL "")
    set(SUPPORTS_LL_VARIANT 1)
else()
    message(FATAL_ERROR "Unknown or unsupported board!")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "-EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=${NUM_ENGINES}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
# specify -MMD -fsycl-link-targets=... instead of -fintelfpga to workaround known issue; lower report quality
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -MMD -fintelfpga -DNUM_ENGINES=${NUM_ENGINES}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsparallel=2 -Xsopt-arg=\"-nocaching\" -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS} -DNUM_ENGINES=${NUM_ENGINES}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#   dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 gzipkernel.cpp gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp -o gzip.fpga_emu
# CMake executes:
#    [compile] dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 -o gzipkernel.cpp.o -c gzipkernel.cpp
#    [compile] dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 -o gzip.cpp.o -c gzip.cpp
#    [compile] dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 -o crc32.cpp.o -c crc32.cpp
#    [compile] dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 -o WriteGzip.cpp.o -c WriteGzip.cpp
#    [compile] dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 -o CompareGzip.cpp.o -c CompareGzip.cpp
#    [link]    dpcpp -fintelfpga gzipkernel.cpp.o gzip.cpp.o crc32.cpp.o WriteGzip.cpp.o CompareGzip.cpp.o -o gzip.fpga_emu
add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

# Low Latency GZIP Variant
if(${SUPPORTS_LL_VARIANT})
    # The compile output is not an executable, but an intermediate compilation result unique to DPC++.
    add_executable(${EMULATOR_TARGET_LL} ${DEVICE_SOURCE_FILE_LL} ${HOST_SOURCE_FILE_LL})
    add_custom_target(fpga_emu_ll DEPENDS ${EMULATOR_TARGET_LL})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus
endif()

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -MMD -fintelfpga -c -DNUM_ENGINES=1 -Xshardware -Xsparallel=2 -Xsopt-arg="-nocaching" -Xsboard=<FPGA_BOARD> -fsycl-link=early gzipkernel.cpp -o gzip_report
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++
add_executable(${FPGA_EARLY_IMAGE} ${DEVICE_SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

# Low Latency GZIP Variant
if(${SUPPORTS_LL_VARIANT})
    # The compile output is not an executable, but an intermediate compilation result unique to DPC++.
    set(FPGA_EARLY_IMAGE_LL ${TARGET_NAME}_ll_report.a)
    add_executable(${FPGA_EARLY_IMAGE_LL} ${DEVICE_SOURCE_FILE_LL})
    add_custom_target(report_ll DEPENDS ${FPGA_EARLY_IMAGE_LL})
    set_target_properties(${FPGA_EARLY_IMAGE_LL} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_LL} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER_LL} -fsycl-link=early")
    # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus
endif()

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -g0 -fintelfpga -DFPGA_EMULATOR -DNUM_ENGINES=1 gzipkernel.cpp gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp -o gzip.fpga_emu
# CMake executes:
#    [compile] dpcpp -MMD -fintelfpga -DNUM_ENGINES=1 -o gzipkernel.cpp.o -c gzipkernel.cpp
#    [compile] dpcpp -MMD -fintelfpga -DNUM_ENGINES=1 -o gzip.cpp.o -c gzip.cpp
#    [compile] dpcpp -MMD -fintelfpga -DNUM_ENGINES=1 -ocrc32.cpp.o -c crc32.cpp
#    [compile] dpcpp -MMD -fintelfpga -DNUM_ENGINES=1 -o -DNUM_ENGINES=1 -o WriteGzip.cpp.o -c WriteGzip.cpp
#    [compile] dpcpp -MMD -fintelfpga -DNUM_ENGINES=1 -o CompareGzip.cpp.o -c CompareGzip.cpp
#    [link]    dpcpp -fintelfpga -Xshardware -Xsparallel=2 -Xsopt-arg="-nocaching" -Xsboard=<FPGA_BOARD> -DNUM_ENGINES=1 -Xsseed=8 gzipkernel.cpp.o gzip.cpp.o crc32.cpp.o WriteGzip.cpp.o CompareGzip.cpp.o -o gzip.fpga_emu
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER} ${HIGH_BW_SEED} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

# Low Latency GZIP Variant
if(${SUPPORTS_LL_VARIANT})
    add_executable(${FPGA_TARGET_LL} EXCLUDE_FROM_ALL ${DEVICE_SOURCE_FILE_LL} ${HOST_SOURCE_FILE_LL})
    add_custom_target(fpga_ll DEPENDS ${FPGA_TARGET})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER_LL} ${LL_SEED} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
    # The -reuse-exe flag enables rapid recompilation of host-only code changes.
    # See DPC++FPGA/GettingStarted/fast_recompile for details.
endif()
