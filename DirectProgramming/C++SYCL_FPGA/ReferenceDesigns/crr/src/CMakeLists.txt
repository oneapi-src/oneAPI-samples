set(TARGET_NAME crr)
set(SOURCE_FILE main.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    set(DEVICE_FLAG "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    string(TOLOWER ${FPGA_DEVICE} FPGA_DEVICE_NAME)
    if(FPGA_DEVICE_NAME MATCHES ".*a10.*" OR FPGA_DEVICE_NAME MATCHES ".*arria10.*")
      set(DEVICE_FLAG "A10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*s10.*" OR FPGA_DEVICE_NAME MATCHES ".*stratix10.*")
      set(DEVICE_FLAG "S10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*agilex.*")
      set(DEVICE_FLAG "Agilex7")
    endif()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Please make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or \
                         -DDEVICE_FLAG=Agilex7.")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Set design parameters according to the selected board
if(DEVICE_FLAG MATCHES "A10")
    # A10 parameters
    set(OUTER_UNROLL_V 1)
    set(INNER_UNROLL_V 64)
    set(OUTER_UNROLL_POW2_V 1)
    set(SEED "-Xsseed=1")
elseif(DEVICE_FLAG MATCHES "S10")
    # S10 parameters
    set(OUTER_UNROLL_V 2)
    set(INNER_UNROLL_V 64)
    set(OUTER_UNROLL_POW2_V 2)
    set(SEED "-Xsseed=3")
elseif(DEVICE_FLAG MATCHES "Agilex7")
    # Agilex 7
    set(OUTER_UNROLL_V 2)
    set(INNER_UNROLL_V 64)
    set(OUTER_UNROLL_POW2_V 2)
    set(SEED "-Xsseed=4")
else()
    message(FATAL_ERROR "Unknown board!")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

message(STATUS "OUTER_UNROLL_V=${OUTER_UNROLL_V}")
message(STATUS "INNER_UNROLL_V=${INNER_UNROLL_V}")
message(STATUS "OUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V}")

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} -Xsdaz -Xsrounding=faithful -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 ${SEED} -Xstarget=${FPGA_DEVICE} -DOUTER_UNROLL_V=${OUTER_UNROLL_V} -DINNER_UNROLL_V=${INNER_UNROLL_V} -DOUTER_UNROLL_POW2_V=${OUTER_UNROLL_POW2_V} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

# Copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)
configure_file("data/small_ordered_inputs.csv" "data/small_ordered_inputs.csv" COPYONLY)

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
