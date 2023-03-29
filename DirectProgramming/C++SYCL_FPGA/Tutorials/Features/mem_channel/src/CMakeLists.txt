set(SOURCE_FILE mem_channel.cpp)
set(TARGET_NAME mem_channel)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET_INTERLEAVING ${TARGET_NAME}_interleaving.fpga)
set(FPGA_TARGET_NO_INTERLEAVING ${TARGET_NAME}_no_interleaving.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA simulator compilation
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

set(NO_INTERLEAVING_HARDWARE_COMPILE_FLAGS "-DNO_INTERLEAVING")
if (FPGA_DEVICE STREQUAL "intel_a10gx_pac:pac_a10")
  string(APPEND NO_INTERLEAVING_HARDWARE_COMPILE_FLAGS " -DTWO_CHANNELS")
elseif (FPGA_DEVICE STREQUAL "intel_s10sx_pac:pac_s10")
  string(APPEND NO_INTERLEAVING_HARDWARE_COMPILE_FLAGS " -DFOUR_CHANNELS")
endif()

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR mem_channel.cpp -o mem_channel.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o mem_channel.cpp.o -c mem_channel.cpp
#    [link]    icpx -fsycl -fintelfpga mem_channel.cpp.o -o mem_channel.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR mem_channel.cpp -o mem_channel.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o mem_channel.cpp.o -c mem_channel.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation  mem_channel.cpp.o -o mem_channel.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early mem_channel.cpp -o mem_channel_report.a
set(FPGA_EARLY_IMAGE_INTERLEAVING ${TARGET_NAME}_interleaving_report.a)
set(FPGA_EARLY_IMAGE_NO_INTERLEAVING ${TARGET_NAME}_no_interleaving_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE_INTERLEAVING} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_INTERLEAVING} PRIVATE ../../../../include)
add_executable(${FPGA_EARLY_IMAGE_NO_INTERLEAVING} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_NO_INTERLEAVING} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE_INTERLEAVING} ${FPGA_EARLY_IMAGE_NO_INTERLEAVING})
set_target_properties(${FPGA_EARLY_IMAGE_INTERLEAVING} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE_INTERLEAVING} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set_target_properties(${FPGA_EARLY_IMAGE_NO_INTERLEAVING} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} ${NO_INTERLEAVING_HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE_NO_INTERLEAVING} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsno-interleaving=default -fsycl-link=early")
#fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> mem_channel.cpp -o mem_channel.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o mem_channel.cpp.o -c mem_channel.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> mem_channel.cpp.o -o mem_channel.fpga
add_executable(${FPGA_TARGET_INTERLEAVING} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_INTERLEAVING} PRIVATE ../../../../include)
add_executable(${FPGA_TARGET_NO_INTERLEAVING} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_NO_INTERLEAVING} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET_INTERLEAVING} ${FPGA_TARGET_NO_INTERLEAVING})
set_target_properties(${FPGA_TARGET_INTERLEAVING} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_INTERLEAVING} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_INTERLEAVING}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
set_target_properties(${FPGA_TARGET_NO_INTERLEAVING} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} ${NO_INTERLEAVING_HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_NO_INTERLEAVING} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsno-interleaving=default -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_NO_INTERLEAVING}")
