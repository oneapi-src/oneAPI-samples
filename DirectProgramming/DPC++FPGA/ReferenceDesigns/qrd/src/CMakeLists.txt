set(TARGET_NAME qrd)
set(SOURCE_FILE qrd_demo.cpp)
set(DEVICE_SOURCE_FILE qrd.cpp)
set(DEVICE_HEADER_FILE qrd.hpp)

set(UNROLLED_LOOP_HEADER UnrolledLoop.hpp)
set(HOST_SOURCE_FILE qrd_demo.cpp)
set(TARGET_NAME qrd)

set(EMULATOR_TARGET ${TARGET_NAME}.prj)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
set(FIXED_ITERATIONS_A10 64)
set(FIXED_ITERATIONS_S10 105)

set(SEED_A10 9)
set(SEED_S10 5)

set(CLOCK_TARGET_A10 360MHz)
set(CLOCK_TARGET_S10 480MHz)

# Set parameter values assuming target is Intel(R) PAC with Intel Arria(R) 10 GX FPGA
set(SELECTED_BOARD ${A10_PAC_BOARD_NAME})
set(FIXED_ITERATIONS ${FIXED_ITERATIONS_A10})
set(CLOCK_TARGET ${CLOCK_TARGET_A10})
set(SEED ${SEED_A10})

if(DEFINED ENV{ROWS_COMPONENT} AND DEFINED ENV{COLS_COMPONENT})
    set(ROWS_COMPONENT $ENV{ROWS_COMPONENT})
    set(COLS_COMPONENT $ENV{COLS_COMPONENT})    
else()
    set(ROWS_COMPONENT ${ROWS_COMPONENT_A10})
    set(COLS_COMPONENT ${COLS_COMPONENT_A10})
endif()

if(DEFINED ENV{COMPLEX})
    set(COMPLEX $ENV{COMPLEX})
else()
    set(COMPLEX 1)
endif()

# Check if target is the Intel(R) PAC D5005 (with Intel Stratix(R) 10 SX FPGA)
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
elseif(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")
elseif(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) D5005 (with Intel Stratix(R) 10 SX FPGA).")
    set(SELECTED_BOARD ${S10_PAC_BOARD_NAME})
    set(FIXED_ITERATIONS ${FIXED_ITERATIONS_S10})
    set(CLOCK_TARGET ${CLOCK_TARGET_S10})
    set(SEED ${SEED_S10})
    set(ROWS_COMPONENT ${ROWS_COMPONENT_S10})
    set(COLS_COMPONENT ${COLS_COMPONENT_S10})
else()
    message(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
endif()

set(HARDWARE_COMPILE_FLAGS -fbracket-depth=512 -fintelfpga -c -fno-fast-math -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS} -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})
set(FINAL_LINK_FLAGS -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})

set(EMULATOR_COMPILE_FLAGS "-fbracket-depth=512 -fintelfpga -fno-finite-math-only -DFPGA_EMULATOR -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(EMULATOR_LINK_FLAGS -fintelfpga )

# FPGA emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${WIN_EMULATOR_COMPILE_FLAGS} /F 2097152 /EHsc ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
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
    set(SEED 31)
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
    set(SEED 5)
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™ parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 105)
    set(CLOCK_TARGET 480MHz)
    set(SEED 5)
else()
    message(FATAL_ERROR "Unknown board!")
endif()

message(STATUS "ROWS_COMPONENT=${ROWS_COMPONENT}")
message(STATUS "COLS_COMPONENT=${COLS_COMPONENT}")
message(STATUS "FIXED_ITERATIONS=${FIXED_ITERATIONS}")
message(STATUS "SEED=${SEED}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -DFPGA_EMULATOR -qactypes")
set(EMULATOR_LINK_FLAGS "-fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -fno-fast-math -Xsfp-relaxed -qactypes")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} ${USER_HARDWARE_FLAGS} -fno-fast-math -Xsfp-relaxed")
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
