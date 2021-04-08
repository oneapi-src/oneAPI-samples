# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE mvdr_beamforming.cpp)
set(TARGET_NAME mvdr_beamforming)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
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

# check if the BSP has USM host allocations
if(FPGA_BOARD MATCHES ".usm.*")
    set(ENABLE_USM "-DUSM_HOST_ALLOCATIONS")
    message(STATUS "USM host allocations are enabled")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
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


# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -fbracket-depth=512 ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${ENABLE_USM}")
set(SIMULATOR_COMPILE_FLAGS "-fintelfpga -fbracket-depth=512 ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fintelfpga -fbracket-depth=512 -Xssimulation -Xsghdl")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fbracket-depth=512 -fintelfpga ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -fbracket-depth=512 ${PROFILE_FLAG} ${ENABLE_USM} ${SENSOR_SIZE_FLAG} ${NUM_SENSORS_FLAG} ${QRD_MIN_ITERATIONS_FLAG} -Xsparallel=2 -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -fbracket-depth=512 -DFPGA_EMULATOR mvdr_beamforming.cpp -o mvdr_beamforming.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp
#    [link]    dpcpp -fintelfpga mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -fbracket-depth=512 -Xsboard=<FPGA_BOARD> -fsycl-link=early mvdr_beamforming.cpp -o mvdr_beamforming_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set(CMAKE_CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
separate_arguments(CMAKE_CXX_FLAGS_LIST)
set(HARDWARE_LINK_FLAGS_LIST ${HARDWARE_LINK_FLAGS})
separate_arguments(HARDWARE_LINK_FLAGS_LIST)
add_custom_command(OUTPUT ${FPGA_EARLY_IMAGE}
                   COMMAND ${CMAKE_CXX_COMPILER} ${WIN_FLAG} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=early ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${FPGA_EARLY_IMAGE}
                   DEPENDS ${SOURCE_FILE})
                   # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -fbracket-depth=512 -Xssimulation -Xsghdl -Xsboard=<FPGA_BOARD> mvdr_beamforming.cpp -o mvdr_beamforming.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -fbracket-depth=512 -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -fbracket-depth=512 -Xssimulation -Xsghdl -Xsboard=<FPGA_BOARD> mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -fbracket-depth=512 -Xsboard=<FPGA_BOARD> mvdr_beamforming.cpp -o mvdr_beamforming.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -fbracket-depth=512 -o mvdr_beamforming.cpp.o -c mvdr_beamforming.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -fbracket-depth=512 -Xsboard=<FPGA_BOARD> mvdr_beamforming.cpp.o -o mvdr_beamforming.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

