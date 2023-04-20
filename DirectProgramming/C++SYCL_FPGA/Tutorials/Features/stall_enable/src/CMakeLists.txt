set(SOURCE_FILE stall_enable.cpp)
set(STALL_ENABLE_TARGET_NAME stall_enable)
set(EMULATOR_TARGET ${STALL_ENABLE_TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${STALL_ENABLE_TARGET_NAME}.fpga_sim)
set(FPGA_TARGET_STALL_ENABLE ${STALL_ENABLE_TARGET_NAME}.fpga)

set(STALL_FREE_TARGET_NAME stall_free)
set(STALL_FREE_DEFINE -DSTALL_FREE)
set(STALL_FREE_EMULATOR_TARGET ${STALL_FREE_TARGET_NAME}.fpga_emu)
set(STALL_FREE_SIMULATOR_TARGET ${STALL_FREE_TARGET_NAME}.fpga_sim)
set(FPGA_TARGET_STALL_FREE ${STALL_FREE_TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

# Allow disabling of hyper-optimization for S10 and Agilex7
if (${NO_HYPER_OPTIMIZATION})
    set(HYPER_FLAG -Xshyper-optimized-handshaking=off)
endif(${NO_HYPER_OPTIMIZATION})

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
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} ${HYPER_FLAG} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} ${HYPER_FLAG} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} ${HYPER_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${HYPER_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR stall_enable.cpp -o stall_enable.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o stall_enable.cpp.o -c stall_enable.cpp
#    [link]    icpx -fsycl -fintelfpga stall_enable.cpp.o -o stall_enable.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_executable(${STALL_FREE_EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${STALL_FREE_EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${STALL_FREE_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS} ${STALL_FREE_DEFINE}")
set_target_properties(${STALL_FREE_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET} ${STALL_FREE_EMULATOR_TARGET})
###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR triangular_loop.cpp -o triangular_loop.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o triangular_loop.cpp.o -c triangular_loop.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> triangular_loop.cpp.o -o triangular_loop.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_executable(${STALL_FREE_SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${STALL_FREE_SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${STALL_FREE_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STALL_FREE_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET} ${STALL_FREE_SIMULATOR_TARGET})


###############################################################################
### Generate Report
###############################################################################
# To compile manually (Stall Enable):
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early stall_enable.cpp -o stall_enable.a
set(FPGA_EARLY_IMAGE ${STALL_ENABLE_TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

# To compile manually (Stall Free):
#   icpx -DSTALL_FREE -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early stall_enable.cpp -o stall_enable.a
set(FPGA_STALL_FREE_EARLY_IMAGE ${STALL_FREE_TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_STALL_FREE_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_STALL_FREE_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} ${FPGA_STALL_FREE_EARLY_IMAGE})
set_target_properties(${FPGA_STALL_FREE_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${STALL_FREE_DEFINE} ${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_STALL_FREE_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command (Stall Enable):
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> stall_enable.cpp -o stall_enable.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o stall_enable.cpp.o -c stall_enable.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> stall_enable.cpp.o -o stall_enable.fpga
add_executable(${FPGA_TARGET_STALL_ENABLE} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_STALL_ENABLE} PRIVATE ../../../../include)
set_target_properties(${FPGA_TARGET_STALL_ENABLE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_STALL_ENABLE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_STALL_ENABLE}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

# To compile in a single command (Stall Enable):
#   icpx -fsycl -fintelfpga -DSTALL_FREE -Xshardware -Xstarget=<FPGA_DEVICE> stall_enable.cpp -o stall_free.fpga
# CMake executes:
#   [compile] icpx -DSTALL_FREE -fsycl -fintelfpga -o stall_free.cpp.o -c stall_free.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> stall_free.cpp.o -o stall_free.fpga
add_executable(${FPGA_TARGET_STALL_FREE} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_STALL_FREE} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET_STALL_ENABLE} ${FPGA_TARGET_STALL_FREE})
set_target_properties(${FPGA_TARGET_STALL_FREE} PROPERTIES COMPILE_FLAGS "${STALL_FREE_DEFINE} ${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_STALL_FREE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_STALL_FREE}")
