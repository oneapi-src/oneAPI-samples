

# # Intel supported FPGA Boards
# "intel_a10gx_pac:pac_a10"
# "intel_s10sx_pac:pac_s10"


set(TARGET_NAME qri)
set(DEVICE_SOURCE_FILE qri.cpp)
set(DEVICE_HEADER_FILE qri.hpp)
set(UNROLLED_LOOP_HEADER UnrolledLoop.hpp)
set(TUPLE_HEADER Tuple.hpp)
set(STREAMING_QRD_HEADER StreamingQRD.hpp)
set(STREAMING_QRI_HEADER StreamingQRI.hpp)
set(UTILS_HEADER Utils.hpp)
set(MEMORY_TRANSFERS_HEADER MemoryTransfers.hpp)
set(DIM_HEADER QRInversionDim.hpp)
set(HOST_SOURCE_FILE qri_demo.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Set design parameters according to the selected board
if(FPGA_BOARD MATCHES ".*a10.*")
    # A10 parameters
    set(ROWS_COMPONENT 32)
    set(COLS_COMPONENT 32)
    set(COMPLEX 0)
    set(FIXED_ITERATIONS 64)
    set(CLOCK_TARGET 360MHz)
    set(SEED 10)
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(ROWS_COMPONENT 32)
    set(COLS_COMPONENT 32)
    set(COMPLEX 0)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 500MHz)
    set(SEED 5)
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™ parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
    set(SEED 5)
else()
    message(FATAL_ERROR "Unknown board!")
endif()


if(DEFINED ENV{ROWS_COMPONENT} AND DEFINED ENV{COLS_COMPONENT})
    set(ROWS_COMPONENT $ENV{ROWS_COMPONENT})
    set(COLS_COMPONENT $ENV{COLS_COMPONENT})
endif()

if(DEFINED ENV{COMPLEX})
    set(COMPLEX $ENV{COMPLEX})
endif()

if(DEFINED ENV{FIXED_ITERATIONS})
    set(FIXED_ITERATIONS $ENV{FIXED_ITERATIONS})
endif()

message(STATUS "ROWS_COMPONENT=${ROWS_COMPONENT}")
message(STATUS "COLS_COMPONENT=${COLS_COMPONENT}")
message(STATUS "COMPLEX=${COMPLEX}")
message(STATUS "FIXED_ITERATIONS=${FIXED_ITERATIONS}")
message(STATUS "SEED=${SEED}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-I/p/psg/ctools/mpfr/2.4.2/linux64/include/ -I/p/psg/ctools/gmp/6.1.2/linux64/include/ -Wall ${WIN_FLAG} -fbracket-depth=512 -fintelfpga -fno-finite-math-only -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga -L/p/psg/ctools/mpfr/2.4.2/linux64/lib/ -lmpfr")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -fbracket-depth=512 -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -fno-fast-math -Xsfp-relaxed")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS} -fno-fast-math -Xsfp-relaxed")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.