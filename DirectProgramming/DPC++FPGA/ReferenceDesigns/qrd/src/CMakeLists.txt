set(DEVICE_SOURCE_FILE qrd.cpp)
set(DEVICE_HEADER_FILE qrd.hpp)
set(HOST_SOURCE_FILE qrd_demo.cpp)
set(TARGET_NAME qrd)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# Set design parameters according to the selected board
if(FPGA_BOARD STREQUAL "intel_a10gx_pac:pac_a10")
    # A10 parameters
    set(ROWS_COMPONENT 128)
    set(COLS_COMPONENT 128)
    set(FIXED_ITERATIONS 64)
    set(SEED 9)
    set(CLOCK_TARGET 360MHz)
elseif(FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10")
    # S10 parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(SEED 5)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
else()
    message(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -c -fno-fast-math -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS} -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 qrd.cpp qrd_demo.cpp -o qrd.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 -o qrd.cpp.o -c qrd.cpp
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 -o qrd_demo.cpp.o -c qrd_demo.cpp
#    [link]    dpcpp -fintelfpga qrd.cpp.o qrd_demo.cpp.o -o qrd.fpga_emu
add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsclock=360MHz -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=9 -Xsboard=intel_a10gx_pac:pac_a10 -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 -fsycl-link=early qrd.cpp -o qrd_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${DEVICE_SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsclock=360MHz -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=9 -Xsboard=<FPGA_BOARD> -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 qrd.cpp qrd_demo.cpp -o qrd.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -c -fno-fast-math -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 -o qrd.cpp.o -c qrd.cpp
#   [compile] dpcpp -fintelfpga -c -fno-fast-math -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 -o qrd_demo.cpp.o -c qrd_demo.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsclock=360MHz -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=9 -Xsboard=<FPGA_BOARD> -DFIXED_ITERATIONS=64 -DROWS_COMPONENT=128 -DCOLS_COMPONENT=128 qrd.cpp.o qrd_demo.cpp.o -o qrd.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

# Run emulator
add_custom_target(run
                  COMMAND ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET}
                  DEPENDS ${EMULATOR_TARGET})
