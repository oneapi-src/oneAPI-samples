set(SOURCE_FILE mvdr_beamforming.cpp)
set(TARGET_NAME mvdr_beamforming)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

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

# check if the BSP has USM host allocations
if((IS_BSP STREQUAL "0") OR FPGA_DEVICE MATCHES ".usm.*")
    set(ENABLE_USM "-DUSM_HOST_ALLOCATIONS")
    message(STATUS "USM host allocations are enabled")
endif()

# These are Windows-specific flags:
# 1. /EHsc This is a Windows-specific flag that enables exception handling in host code
# 2. /Qactypes Include ac_types headers and link against ac_types emulation libraries
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes")
else()
    set(AC_TYPES_FLAG "-qactypes")
endif()

# Allow the user to enable real IO pipes
# e.g. cmake .. -DREAL_IO_PIPES=1
if(REAL_IO_PIPES)
    set(REAL_IO_PIPES_FLAG "-DREAL_IO_PIPES")
    set(STREAMING_PIPE_WIDTH_FLAG "-DSTREAMING_PIPE_WIDTH=1")
    set(UDP_LINK_FLAGS "-luuid -lopae-c -lpthread")
    message(STATUS "Design is using real IO pipes")

    # the real IO pipes version doesn't work on Windows, so error out
    if(WIN32)
      message(FATAL_ERROR "The real IO pipe design is only supported on Linux")
    endif()
endif()

if(FLAT_COMPILE)
  message(STATUS "Doing a flat compile")
  set(FLAT_COMPILE_FLAG "-Xsbsp-flow=flat")
endif()

# Allow the user to enable hardware profiling
# Profiling can be enabled when running cmake by adding the flag -DPROFILE_HW=1
# e.g. cmake .. -DPROFILE_HW=1
if(PROFILE_HW)
    set(PROFILE_FLAG "-Xsprofile")
endif()

# Allow the user to select a larger matrix size (64 sensors)
# e.g. cmake .. -DLARGE_SENSOR_ARRAY=1
if(LARGE_SENSOR_ARRAY)
    set(SENSOR_SIZE_FLAG "-DLARGE_SENSOR_ARRAY")
endif()

# Allow an arbitrary size sensor array (hardware runs will not work due to
# lack of expected data)
# e.g. cmake .. -DNUM_SENSORS=96
if(NUM_SENSORS)
    set(NUM_SENSORS_FLAG "-DNUM_SENSORS=${NUM_SENSORS}")
endif()

# Allow the user to set the min iterations for the QRD kernel
# e.g. cmake .. -DQRD_MIN_ITERATIONS=85
if(QRD_MIN_ITERATIONS)
    set(QRD_MIN_ITERATIONS_FLAG "-DQRD_MIN_ITERATIONS=${QRD_MIN_ITERATIONS}")
endif()

# Allow the user to set the streaming pipe width for the input/output pipes
# e.g. cmake .. -DSTREAMING_PIPE_WIDTH=2
if(STREAMING_PIPE_WIDTH)
    set(STREAMING_PIPE_WIDTH_FLAG "-DSTREAMING_PIPE_WIDTH=${STREAMING_PIPE_WIDTH}")
endif()


# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -fbracket-depth=512 ${AC_TYPES_FLAG} ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} ${STREAMING_PIPE_WIDTH_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} ${ENABLE_USM}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -fbracket-depth=512 ${AC_TYPES_FLAG} ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} ${STREAMING_PIPE_WIDTH_FLAG} -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Wall -fbracket-depth=512 ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} ${REAL_IO_PIPES_FLAG} ${STREAMING_PIPE_WIDTH_FLAG} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${UDP_LINK_FLAGS} ${AC_TYPES_FLAG} -Xssimulation -Xsghdl")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga ${WIN_FLAG} -fbracket-depth=512 ${AC_TYPES_FLAG} ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} ${REAL_IO_PIPES_FLAG} ${STREAMING_PIPE_WIDTH_FLAG} -DFPGA_HARDWARE")
set(REPORT_LINK_FLAGS "-fsycl -fintelfpga -Wall -Xshardware -fbracket-depth=512 ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} ${REAL_IO_PIPES_FLAG} ${STREAMING_PIPE_WIDTH_FLAG} ${PROFILE_FLAG} -Xsparallel=2 -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${UDP_LINK_FLAGS}")
set(HARDWARE_LINK_FLAGS "${REPORT_LINK_FLAGS} ${AC_TYPES_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -fbracket-depth=512 -qactypes -DFPGA_EMULATOR mvdr_beamforming.cpp -o mvdr_beamforming.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp
#    [link]    icpx -fsycl -fintelfpga mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -fbracket-depth=512 -qactypes -Xstarget=<FPGA_DEVICE> -fsycl-link=early mvdr_beamforming.cpp -o mvdr_beamforming_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -fbracket-depth=512 -qactypes -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> mvdr_beamforming.cpp -o mvdr_beamforming.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -fbracket-depth=512 -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp -FPGA_SIMULATOR
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -fbracket-depth=512 -Xstarget=<FPGA_DEVICE> -Xsghdl mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -fbracket-depth=512 -qactypes -Xstarget=<FPGA_DEVICE> mvdr_beamforming.cpp -o mvdr_beamforming.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -fbracket-depth=512 -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -fbracket-depth=512 -Xstarget=<FPGA_DEVICE> mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.


###############################################################################
# UDP Loopback test
###############################################################################
set(UDP_LOOPBACK_TARGET udp_loopback_test.fpga)
set(UDP_LOOPBACK_COMPILE_FLAGS "-Wall -fsycl -fintelfpga")
set(UDP_LOOPBACK_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${FLAT_COMPILE_FLAG} ${USER_HARDWARE_FLAGS} ${UDP_LINK_FLAGS}")
add_executable(${UDP_LOOPBACK_TARGET} EXCLUDE_FROM_ALL udp_loopback_test.cpp)
add_custom_target(udp_loopback_test DEPENDS ${UDP_LOOPBACK_TARGET})
set_target_properties(${UDP_LOOPBACK_TARGET} PROPERTIES COMPILE_FLAGS "${UDP_LOOPBACK_COMPILE_FLAGS}")
set_target_properties(${UDP_LOOPBACK_TARGET} PROPERTIES LINK_FLAGS "${UDP_LOOPBACK_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${UDP_LOOPBACK_TARGET}")

