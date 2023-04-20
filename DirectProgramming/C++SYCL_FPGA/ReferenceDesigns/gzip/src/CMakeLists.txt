set(TARGET_NAME gzip)


if(LOW_LATENCY)
    # Compile the low latency version of the design
    message(STATUS "Compiling the Low Latency variant of the design")
    set(SOURCE_FILE gzip_ll.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)
    set(DEVICE_SOURCE_FILE gzipkernel_ll.cpp)
    set(DEVICE_HEADER_FILE gzipkernel_ll.hpp)
else()
    # Compile the high bandwidth version of the design
    message(STATUS "Compiling the High Bandwidth variation of the design")
    set(SOURCE_FILE gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)
    set(DEVICE_SOURCE_FILE gzipkernel.cpp)
    set(DEVICE_HEADER_FILE gzipkernel.hpp)
endif()

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

    set(IS_BSP "0")
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

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1" OR FPGA_DEVICE MATCHES ".*pac_a10.*|.*pac_s10.*")
        set(IS_BSP "1")
    else()
        set(IS_BSP "0")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number, so USM will be enabled by default.")
        message(STATUS "If the target is actually a BSP that does not support USM, run cmake with -DIS_BSP=1.")
    endif()
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or -DDEVICE_FLAG=Agilex7.")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Set design parameters according to the selected chip
if(DEVICE_FLAG MATCHES "A10")
    # A10 parameters
    set(NUM_ENGINES 1)
    if(DEFINED LOW_LATENCY)
        set(SEED "-Xsseed=4")
        set(NUM_REORDER "")
    else()
        set(SEED "-Xsseed=4")
        set(NUM_REORDER "")
    endif()
elseif(DEVICE_FLAG MATCHES "S10")
    # S10 parameters
    set(NUM_ENGINES 2)
    if(DEFINED LOW_LATENCY)
        set(SEED "-Xsseed=2")
        set(NUM_REORDER "")
    else()
        set(SEED "-Xsseed=10")
        # For the High Bandwidth variant, specify 6 reordering units to improve global memory read bandwidth across 4 channels of DDR.
        # For Low Latency variant this is not necessary since only one channel of global memory is used (host memory).
        set(NUM_REORDER "-Xsnum-reorder=6")
    endif()
elseif(DEVICE_FLAG MATCHES "Agilex7")
    # Agilex 7
    set(NUM_ENGINES 2)
    if(DEFINED LOW_LATENCY)
        set(SEED "-Xsseed=1")
        set(NUM_REORDER "")
    else()
        set(SEED "-Xsseed=8")
        # For the High Bandwidth variant, specify 6 reordering units to improve global memory read bandwidth across 4 channels of DDR.
        # For Low Latency variant this is not necessary since only one channel of global memory is used (host memory).
        set(NUM_REORDER "-Xsnum-reorder=6")
    endif()
else()
    set(NUM_ENGINES 1)
    set(SEED "-Xsseed=1")
    set(NUM_REORDER "")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

# Presence of USM host allocations (and whether to turn on enable the low-latency target) is detected automatically by
# looking at the name of the BSP, or manually by the user when running CMake.
# E.g., cmake .. -DUSM_HOST_ALLOCATIONS_ENABLED=1
if((IS_BSP STREQUAL "1") AND LOW_LATENCY AND NOT FPGA_DEVICE MATCHES ".usm.*" AND (NOT DEFINED USM_HOST_ALLOCATIONS_ENABLED OR USM_HOST_ALLOCATIONS_ENABLED STREQUAL "0"))
    # Low latency design requires USM, so error out
    message(FATAL_ERROR "Error: The Low Latency variant of the design requires USM host allocations")
endif()

if(USM_HOST_ALLOCATIONS_ENABLED)
    message(STATUS "USM_HOST_ALLOCATIONS_ENABLED set manually!")
endif()

message(STATUS "NUM_ENGINES=${NUM_ENGINES}")
message(STATUS "SEED=${SEED}")
message(STATUS "NUM_REORDER=${NUM_REORDER}")

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DNUM_ENGINES=${NUM_ENGINES} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga -DNUM_ENGINES=${NUM_ENGINES}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DNUM_ENGINES=${NUM_ENGINES} -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} -DNUM_ENGINES=${NUM_ENGINES} ${USER_SIMULATOR_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DNUM_ENGINES=${NUM_ENGINES} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsparallel=2 -Xsopt-arg=\"-nocaching\" -Xstarget=${FPGA_DEVICE} -DNUM_ENGINES=${NUM_ENGINES} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER} ${SEED} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} ${NUM_REORDER} ${SEED} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.



