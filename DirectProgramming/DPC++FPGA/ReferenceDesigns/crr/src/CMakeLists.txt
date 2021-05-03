set(SOURCE_FILE main.cpp)
set(TARGET_NAME crr)
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

# Set design parameters according to the selected board
if(FPGA_BOARD STREQUAL "intel_a10gx_pac:pac_a10")
    # A10 parameters
    set(OUTER_UNROLL 1)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 1)
    set(SEED 1)
elseif(FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10")
    # S10 parameters
    set(OUTER_UNROLL 2)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 2)
    set(SEED 4)
else()
    message(FATAL_ERROR "Unknown or unsupported board!")
endif()

# Flags
# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -c -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS} -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

# Copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 main.cpp -o crr.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 -o main.cpp.o -c main.cpp
#    [link]    dpcpp -fintelfpga main.cpp.o -o crr.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=1 -Xsboard=<FPGA_BOARD> -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 -fsycl-link=early main.cpp -o crr_report.a
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
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=1 -Xsboard=<FPGA_BOARD> -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 main.cpp -o crr.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -c -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 -o main.cpp.o -c main.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=1 -Xsboard=<FPGA_BOARD> -DOUTER_UNROLL=1 -DINNER_UNROLL=64 -DOUTER_UNROLL_POW2=1 main.cpp.o -o crr.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

# Run emulator
add_custom_target(run
                  COMMAND ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} ${CMAKE_SOURCE_DIR}/src/data/ordered_inputs.csv
                          -o=${CMAKE_SOURCE_DIR}/src/data/ordered_output.csv
                  DEPENDS ${EMULATOR_TARGET})

