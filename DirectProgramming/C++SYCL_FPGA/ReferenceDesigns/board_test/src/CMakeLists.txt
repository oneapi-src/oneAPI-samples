# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(TARGET_NAME board_test)
set(SOURCE_FILE board_test.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

# Check if the target is a BSP
if(NOT FPGA_DEVICE MATCHES ".*:.*")
    message(STATUS "This sample is made to target BSPs as this is a benchmarking sample.")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "/EHsc /Qactypes /Wall")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "/Qactypes")
else()
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "-qactypes -Wformat-security -Werror=format-security -Wall")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "")
endif()

set(SEED "-Xsseed=2")
if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -DFPGA_EMULATOR -Wformat-security -Werror=format-security -Wall")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -DFPGA_HARDWARE")
# By default oneAPI compiler burst interleaves across same memory type,
# -Xsno-interleaving is used to disable burst interleaving and test each memory bank independently
# Refer to https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/disabl-burst-int.html for more information
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xshardware -Xsno-interleaving=default ${SEED} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR board_test.cpp -o board_test.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o board_test.cpp.o -c board_test.cpp
#    [link]    icpx -fsycl -fintelfpga board_test.cpp.o -o board_test.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -Xsno-interleaving=default -fsycl-link=early board_test.cpp -o board_test.a
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -Xsno-interleaving=default board_test.cpp -o board_test.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o board_test.cpp.o -c board_test.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -Xsno-interleaving=default board_test.cpp.o -o board_test.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

