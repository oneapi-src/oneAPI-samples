set(TARGET_NAME crr)
set(SOURCE_FILE main.cpp)
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
    set(OUTER_UNROLL 1)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 1)
    set(SEED 1)
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(OUTER_UNROLL 2)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 2)
    set(SEED 4)
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™
    set(OUTER_UNROLL 2)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 2)
    set(SEED 4)
else()
    message(FATAL_ERROR "Unknown board!")
endif()

message(STATUS "OUTER_UNROLL=${OUTER_UNROLL}")
message(STATUS "INNER_UNROLL=${INNER_UNROLL}")
message(STATUS "OUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=2 -Xsboard=${FPGA_BOARD} -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

# Copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
