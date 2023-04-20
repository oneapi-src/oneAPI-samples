set(SOURCE_FILE buffered_host_streaming.cpp)
set(TARGET_NAME buffered_host_streaming)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
    set(IS_BSP "0")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1" OR FPGA_DEVICE MATCHES ".*pac_a10.*|.*pac_s10.*")
        set(IS_BSP "1")
    else()
        set(IS_BSP "0")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number, so USM will be enabled by default.")
        message(STATUS "If the target is actually a BSP that does not support USM, run cmake with -DIS_BSP=1.")
    endif()
endif()

# this tutorial requires USM host allocations. Check the BSP name (which should contain the text 'usm')
# to ensure the BSP has the required support. Allow the user to define USM_HOST_ALLOCATIONS_ENABLED
# to override this check (e.g., cmake .. -DUSM_HOST_ALLOCATIONS_ENABLED=1)
if((IS_BSP STREQUAL "1") AND (NOT FPGA_DEVICE MATCHES ".usm.*") AND (NOT DEFINED USM_HOST_ALLOCATIONS_ENABLED OR USM_HOST_ALLOCATIONS_ENABLED STREQUAL "0"))
    message(FATAL_ERROR "ERROR: This tutorial requires a BSP that has USM host allocations enabled.")
endif()

if (USM_HOST_ALLOCATIONS_ENABLED)
    message(STATUS "USM_HOST_ALLOCATIONS_ENABLED set manually!")
endif()

if(UNIX)
    set(THREAD_FLAG "-lpthread")
else() # Windows
    set(WIN_FLAG "/EHsc") # This is a Windows-specific flag that enables exception handling in host code
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${THREAD_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${THREAD_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${THREAD_FLAG} -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR buffered_host_streaming.cpp -o buffered_host_streaming.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o buffered_host_streaming.cpp.o -c buffered_host_streaming.cpp
#    [link]    icpx -fsycl -fintelfpga buffered_host_streaming.cpp.o -o buffered_host_streaming.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE}) # CMake automatically adds #include'd headers to the dependency list
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR <file>.cpp -o <file>.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> <file>.cpp.o -o <file>.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early buffered_host_streaming.cpp -o buffered_host_streaming_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
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
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> buffered_host_streaming.cpp -o buffered_host_streaming.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o buffered_host_streaming.cpp.o -c buffered_host_streaming.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> buffered_host_streaming.cpp.o -o buffered_host_streaming.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

