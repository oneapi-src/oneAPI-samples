set(SOURCE_FILE hostpipes.cpp)
set(TARGET_NAME hostpipes)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA device selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_s10sx_pac:pac_s10_usm")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA device ${FPGA_DEVICE} (Intel(R) PAC with Intel Stratix(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA device ${FPGA_DEVICE}")
endif()

# this tutorial requires USM host allocations. Check the BSP name (which should contain the text 'usm')
# to ensure the BSP has the required support. Allow the user to define USM_HOST_ALLOCATIONS_ENABLED
# to override this check (e.g., cmake .. -DUSM_HOST_ALLOCATIONS_ENABLED=1)
if(NOT FPGA_DEVICE MATCHES ".usm.*" AND (NOT DEFINED USM_HOST_ALLOCATIONS_ENABLED OR USM_HOST_ALLOCATIONS_ENABLED STREQUAL "0"))
    message(FATAL_ERROR "ERROR: This tutorial requires a BSP that has USM host allocations enabled.")
endif()

if (USM_HOST_ALLOCATIONS_ENABLED)
    message(STATUS "USM_HOST_ALLOCATIONS_ENABLED set manually!")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# This is for finding the install path to the necessary include files
if(DEFINED ENV{INTELFPGAOCLSDKROOT})
   set(SDK_ROOT_PATH $ENV{INTELFPGAOCLSDKROOT})
else()
   message(FATAL_ERROR "The INTELFPGAOCLSDKROOT environment variable must be defined for this tutorial to compile.")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR -I${SDK_ROOT_PATH}/include -I${SDK_ROOT_PATH}/include/sycl/ext/intel/prototype")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} -fintelfpga -I${SDK_ROOT_PATH}/include -I${SDK_ROOT_PATH}/include/sycl/ext/intel/prototype")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsboard=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fintelfpga -DFPGA_EMULATOR hostpipes.cpp -o hostpipes.fpga_emu
# CMake executes:
#    [compile] icpx -fintelfpga -DFPGA_EMULATOR -o hostpipes.cpp.o -c hostpipes.cpp
#    [link]    icpx -fintelfpga hostpipes.cpp.o -o hostpipes.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fintelfpga -Xshardware -Xsboard=<FPGA_DEVICE> -fsycl-link=early hostpipes.cpp -o hostpipes_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> hostpipes.cpp -o hostpipes.fpga
# CMake executes:
#   [compile] icpx -fintelfpga -o hostpipes.cpp.o -c hostpipes.cpp
#   [link]    icpx -fintelfpga -Xshardware -Xsboard=<FPGA_DEVICE> hostpipes.cpp.o -o hostpipes.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
