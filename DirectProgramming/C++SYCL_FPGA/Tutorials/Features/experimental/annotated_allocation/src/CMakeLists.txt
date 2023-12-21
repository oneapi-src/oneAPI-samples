set(ANNOTATED_ALLOC_SOURCE_FILE annotated_alloc.cpp)
set(ANNOTATED_ALLOC_TARGET_NAME annotated_alloc)
set(ANNOTATED_ALLOC_EMULATOR_TARGET ${ANNOTATED_ALLOC_TARGET_NAME}.fpga_emu)
set(ANNOTATED_ALLOC_SIMULATOR_TARGET ${ANNOTATED_ALLOC_TARGET_NAME}.fpga_sim)
set(ANNOTATED_ALLOC_FPGA_TARGET ${ANNOTATED_ALLOC_TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA family: ${FPGA_DEVICE} \
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1")
        set(IS_BSP "1")
    else()
        set(IS_BSP "0")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number.")
        message(STATUS "If the target is actually a BSP, run cmake with -DIS_BSP=1.")
    endif()
endif()

if((IS_BSP STREQUAL "1"))
    message(FATAL_ERROR "ERROR: This tutorial is only supported in the SYCL HLS flow and therefore does not support BSPs as a target.")
endif()


# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
include_directories(${SDK_ROOT_PATH}/include/)
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR annotated_alloc.cpp -o annotated_alloc.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o annotated_alloc.cpp.o -c annotated_alloc.cpp
#    [link]    icpx -fsycl -fintelfpga annotated_alloc.cpp.o -o annotated_alloc.fpga_emu
add_executable(${ANNOTATED_ALLOC_EMULATOR_TARGET} ${ANNOTATED_ALLOC_SOURCE_FILE})
target_include_directories(${ANNOTATED_ALLOC_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${ANNOTATED_ALLOC_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${ANNOTATED_ALLOC_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_custom_target(fpga_emu DEPENDS ${ANNOTATED_ALLOC_EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> annotated_alloc.cpp -o annotated_alloc.fpga_sim
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o annotated_alloc.cpp.o -c annotated_alloc.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> annotated_alloc.cpp.o -o annotated_alloc.fpga_sim
add_executable(${ANNOTATED_ALLOC_SIMULATOR_TARGET} ${ANNOTATED_ALLOC_SOURCE_FILE})
target_include_directories(${ANNOTATED_ALLOC_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${ANNOTATED_ALLOC_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${ANNOTATED_ALLOC_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_custom_target(fpga_sim DEPENDS ${ANNOTATED_ALLOC_SIMULATOR_TARGET})

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> annotated_alloc.cpp -o annotated_alloc.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o annotated_alloc.cpp.o -c annotated_alloc.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> annotated_alloc.cpp.o -o annotated_alloc.fpga
add_executable(${ANNOTATED_ALLOC_FPGA_TARGET} ${ANNOTATED_ALLOC_SOURCE_FILE})
target_include_directories(${ANNOTATED_ALLOC_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${ANNOTATED_ALLOC_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${ANNOTATED_ALLOC_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${ANNOTATED_ALLOC_FPGA_TARGET}")

add_custom_target(fpga DEPENDS ${ANNOTATED_ALLOC_FPGA_TARGET})
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early annotated_alloc.cpp -o _report.a
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
set(ANNOTATED_ALLOC_FPGA_EARLY_IMAGE ${ANNOTATED_ALLOC_TARGET_NAME}_report.a)
add_executable(${ANNOTATED_ALLOC_FPGA_EARLY_IMAGE} ${ANNOTATED_ALLOC_SOURCE_FILE})
target_include_directories(${ANNOTATED_ALLOC_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${ANNOTATED_ALLOC_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${ANNOTATED_ALLOC_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

add_custom_target(report DEPENDS ${ANNOTATED_ALLOC_FPGA_EARLY_IMAGE})
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®
