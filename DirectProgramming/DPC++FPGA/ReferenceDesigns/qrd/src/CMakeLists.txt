set(TARGET_NAME qrd)
set(SOURCE_FILE qrd_demo.cpp)
set(DEVICE_SOURCE_FILE qrd.cpp)
set(DEVICE_HEADER_FILE qrd.hpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
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

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Set design parameters according to the selected board
if(FPGA_BOARD MATCHES ".*a10.*")
    # A10 parameters
    set(ROWS_COMPONENT 128)
    set(COLS_COMPONENT 128)
    set(FIXED_ITERATIONS 64)
    set(CLOCK_TARGET 360MHz)
    set(SEED "-Xsseed=5")
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
    set(SEED "-Xsseed=5")
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™ parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
    set(SEED "-Xsseed=5")
else()
    message(FATAL_ERROR "Unknown board!")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

message(STATUS "ROWS_COMPONENT=${ROWS_COMPONENT}")
message(STATUS "COLS_COMPONENT=${COLS_COMPONENT}")
message(STATUS "FIXED_ITERATIONS=${FIXED_ITERATIONS}")
message(STATUS "SEED=${SEED}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -fno-fast-math -Xsfp-relaxed")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 ${SEED} -Xsboard=${FPGA_BOARD} -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} ${USER_HARDWARE_FLAGS} -fno-fast-math -Xsfp-relaxed")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
