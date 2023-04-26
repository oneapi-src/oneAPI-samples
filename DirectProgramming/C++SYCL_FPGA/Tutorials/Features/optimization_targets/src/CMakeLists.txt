set(SOURCE_FILE optimization_targets.cpp)
set(TARGET_NAME_NO_CONTROL no_control)
set(TARGET_NAME_MINIMUM_LATENCY minimum_latency)
set(TARGET_NAME_MANUAL_REVERT manual_revert)
# Emulator target
set(EMULATOR_TARGET_NO_CONTROL ${TARGET_NAME_NO_CONTROL}.fpga_emu)
# Simulator targets
set(SIMULATOR_TARGET_NO_CONTROL ${TARGET_NAME_NO_CONTROL}.fpga_sim)
set(SIMULATOR_TARGET_MINIMUM_LATENCY ${TARGET_NAME_MINIMUM_LATENCY}.fpga_sim)
set(SIMULATOR_TARGET_MANUAL_REVERT ${TARGET_NAME_MANUAL_REVERT}.fpga_sim)
# FPGA Report targets
set(FPGA_EARLY_IMAGE_NO_CONTROL ${TARGET_NAME_NO_CONTROL}_report.a)
set(FPGA_EARLY_IMAGE_MINIMUM_LATENCY ${TARGET_NAME_MINIMUM_LATENCY}_report.a)
set(FPGA_EARLY_IMAGE_MANUAL_REVERT ${TARGET_NAME_MANUAL_REVERT}_report.a)
# FPGA Hardware targets
set(FPGA_TARGET_NO_CONTROL ${TARGET_NAME_NO_CONTROL}.fpga)
set(FPGA_TARGET_MINIMUM_LATENCY ${TARGET_NAME_MINIMUM_LATENCY}.fpga)
set(FPGA_TARGET_MANUAL_REVERT ${TARGET_NAME_MANUAL_REVERT}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    set(DEVICE_FLAG "Agilex7")
    set(MANUAL_REVERT_FLAGS "-Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    string(TOLOWER ${FPGA_DEVICE} FPGA_DEVICE_NAME)
    if(FPGA_DEVICE_NAME MATCHES ".*a10*" OR FPGA_DEVICE_NAME MATCHES ".*arria10.*")
        set(DEVICE_FLAG "A10")
        set(MANUAL_REVERT_FLAGS "-Xssfc-exit-fifo-type=default")
    elseif(FPGA_DEVICE_NAME MATCHES ".*s10*" OR FPGA_DEVICE_NAME MATCHES ".*stratix10.*")
        set(DEVICE_FLAG "S10")
        set(MANUAL_REVERT_FLAGS "-Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default")
    elseif(FPGA_DEVICE_NAME MATCHES ".*agilex*")
        set(DEVICE_FLAG "Agilex7")
        set(MANUAL_REVERT_FLAGS "-Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default")
    endif()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Please make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or \
                         -DDEVICE_FLAG=Agilex7.")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga -DFPGA_EMULATOR -D${DEVICE_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga -DFPGA_SIMULATOR -D${DEVICE_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} -D${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga -DFPGA_HARDWARE -D${DEVICE_FLAG}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} -D${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR optimization_targets.cpp -o no_control.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o no_control.cpp.o -c optimization_targets.cpp
#    [link]    icpx -fsycl -fintelfpga no_control.cpp.o -o no_control.fpga_emu
add_executable(${EMULATOR_TARGET_NO_CONTROL} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET_NO_CONTROL} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET_NO_CONTROL} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET_NO_CONTROL} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET_NO_CONTROL})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early optimization_targets.cpp -o no_control.a
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early optimization_targets.cpp -Xsoptimize=latency -o minimum_latency.a
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early optimization_targets.cpp -Xsoptimize=latency -DMANUAL_REVERT -Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default -o manual_revert.a
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE_NO_CONTROL} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_NO_CONTROL} PRIVATE ../../../../include)
add_executable(${FPGA_EARLY_IMAGE_MINIMUM_LATENCY} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_MINIMUM_LATENCY} PRIVATE ../../../../include)
add_executable(${FPGA_EARLY_IMAGE_MANUAL_REVERT} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_MANUAL_REVERT} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE_NO_CONTROL} ${FPGA_EARLY_IMAGE_MINIMUM_LATENCY} ${FPGA_EARLY_IMAGE_MANUAL_REVERT})
set_target_properties(${FPGA_EARLY_IMAGE_NO_CONTROL} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE_NO_CONTROL} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
set_target_properties(${FPGA_EARLY_IMAGE_MINIMUM_LATENCY} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE_MINIMUM_LATENCY} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early -Xsoptimize=latency")
set_target_properties(${FPGA_EARLY_IMAGE_MANUAL_REVERT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DMANUAL_REVERT")
set_target_properties(${FPGA_EARLY_IMAGE_MANUAL_REVERT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early -Xsoptimize=latency ${MANUAL_REVERT_FLAGS}")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR optimization_targets.cpp -o no_control.fpga_sim
#    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR optimization_targets.cpp -Xsoptimize=latency -o minimum_latency.fpga_sim
#    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR optimization_targets.cpp -Xsoptimize=latency -DMANUAL_REVERT -Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default -o manual_revert.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o no_control.cpp.o -c optimization_targets.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> no_control.cpp.o -o no_control.fpga_sim
#    [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o optimization_targets.cpp.o -c optimization_targets.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> optimization_targets.cpp.o -Xsoptimize=latency -o minimum_latency.fpga_sim
#    [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o manual_revert.cpp.o -c optimization_targets.cpp -DMANUAL_REVERT
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> manual_revert.cpp.o -Xsoptimize=latency -Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default -o manual_revert.fpga_sim
add_executable(${SIMULATOR_TARGET_NO_CONTROL} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET_NO_CONTROL} PRIVATE ../../../../include)
add_executable(${SIMULATOR_TARGET_MINIMUM_LATENCY} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET_MINIMUM_LATENCY} PRIVATE ../../../../include)
add_executable(${SIMULATOR_TARGET_MANUAL_REVERT} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET_MANUAL_REVERT} PRIVATE ../../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET_NO_CONTROL} ${SIMULATOR_TARGET_MINIMUM_LATENCY} ${SIMULATOR_TARGET_MANUAL_REVERT})
set_target_properties(${SIMULATOR_TARGET_NO_CONTROL} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET_NO_CONTROL} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
set_target_properties(${SIMULATOR_TARGET_MINIMUM_LATENCY} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET_MINIMUM_LATENCY} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -Xsoptimize=latency")
set_target_properties(${SIMULATOR_TARGET_MANUAL_REVERT} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS} -DMANUAL_REVERT")
set_target_properties(${SIMULATOR_TARGET_MANUAL_REVERT} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -Xsoptimize=latency ${MANUAL_REVERT_FLAGS}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> optimization_targets.cpp -o no_control.fpga
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> optimization_targets.cpp -Xsoptimize=latency -o minimum_latency.fpga
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> optimization_targets.cpp -Xsoptimize=latency -DMANUAL_REVERT -Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default -o manual_revert.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o no_control.cpp.o -c optimization_targets.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> no_control.cpp.o -o no_control.fpga
#   [compile] icpx -fsycl -fintelfpga -o optimization_targets.cpp.o -c optimization_targets.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> optimization_targets.cpp.o -Xsoptimize=latency -o minimum_latency.fpga
#   [compile] icpx -fsycl -fintelfpga -o manual_revert.cpp.o -c optimization_targets.cpp -DMANUAL_REVERT
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> manual_revert.cpp.o -Xsoptimize=latency -Xshyper-optimized-handshaking=on -Xssfc-exit-fifo-type=default -o manual_revert.fpga
add_executable(${FPGA_TARGET_NO_CONTROL} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_NO_CONTROL} PRIVATE ../../../../include)
add_executable(${FPGA_TARGET_MINIMUM_LATENCY} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_MINIMUM_LATENCY} PRIVATE ../../../../include)
add_executable(${FPGA_TARGET_MANUAL_REVERT} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_MANUAL_REVERT} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET_NO_CONTROL} ${FPGA_TARGET_MINIMUM_LATENCY} ${FPGA_TARGET_MANUAL_REVERT})
set_target_properties(${FPGA_TARGET_NO_CONTROL} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_NO_CONTROL} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_NO_CONTROL}")
set_target_properties(${FPGA_TARGET_MINIMUM_LATENCY} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET_MINIMUM_LATENCY} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_MINIMUM_LATENCY} -Xsoptimize=latency")
set_target_properties(${FPGA_TARGET_MANUAL_REVERT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DMANUAL_REVERT")
set_target_properties(${FPGA_TARGET_MANUAL_REVERT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_MANUAL_REVERT} -Xsoptimize=latency ${MANUAL_REVERT_FLAGS}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
