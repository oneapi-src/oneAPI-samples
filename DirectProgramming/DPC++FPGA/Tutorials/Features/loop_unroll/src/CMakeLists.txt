# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE loop_unroll.cpp)
set(TARGET_NAME loop_unroll)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-Wall -fintelfpga -Xssimulation -DFPGA_SIMULATOR ${USER_SIMULATOR_FLAGS}")
set(SIMULATOR_LINK_FLAGS "-fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} ${USER_SIMULATOR_FLAGS}")
# use cmake -D USER_SIMULATOR_FLAGS=<flags> to set extra flags for FPGA simulator compilation
set(HARDWARE_COMPILE_FLAGS "-Wall -fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR loop_unroll.cpp -o loop_unroll.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o loop_unroll.cpp.o -c loop_unroll.cpp
#    [link]    dpcpp -fintelfpga loop_unroll.cpp.o -o loop_unroll.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR loop_unroll.cpp -o loop_unroll.fpga_sim
# CMake executes:
#    [compile] dpcpp -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o loop_unroll.cpp.o -c loop_unroll.cpp
#    [link]    dpcpp -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> loop_unroll.cpp.o -o loop_unroll.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early loop_unroll.cpp -o loop_unroll_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> loop_unroll.cpp -o loop_unroll.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o loop_unroll.cpp.o -c loop_unroll.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> loop_unroll.cpp.o -o loop_unroll.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
