set(TARGET_NAME merge_sort)
set(SOURCE_FILE main.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")

    set(IS_BSP "0")
    set(BSP_FLAG "")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1" OR FPGA_DEVICE MATCHES ".*pac_a10.*|.*pac_s10.*")
        set(IS_BSP "1")
        set(BSP_FLAG "-DIS_BSP")
    else()
        set(IS_BSP "0")
        set(BSP_FLAG "")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number, so the IS_BSP macro will not be passed to your C++ code and USM will be enabled by default.")
        message(STATUS "If the target is actually a BSP, run cmake with -DIS_BSP=1 to pass the IS_BSP macro to your C++ code and USM checks are performed.")
    endif()
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# check if the BSP has USM host allocations or manually enable using host allocations
# e.g. cmake .. -DUSE_USM_HOST_ALLOCATIONS=1
if((IS_BSP STREQUAL "0") OR FPGA_DEVICE MATCHES ".*usm.*" OR DEFINED USE_USM_HOST_ALLOCATIONS)
    set(ENABLE_USM "-DUSM_HOST_ALLOCATIONS")
    message(STATUS "USM host allocations are enabled")
endif()

# Allow the user to enable hardware profiling
# Profiling can be enabled when running cmake by adding the flag -DPROFILE_HW=1
# e.g. cmake .. -DPROFILE_HW=1
if(PROFILE_HW)
    set(PROFILE_FLAG "-Xsprofile")
endif()

# Select the number of merge units to use. Must be a power of 2.
# e.g. cmake .. -DNUM_MERGE_UNITS=16
if(MERGE_UNITS)
    set(MERGE_UNITS_FLAG "-DMERGE_UNITS=${MERGE_UNITS}")
    message(STATUS "Number of merge units explicitly set to ${MERGE_UNITS}")
endif()

# Select the number of elements each merge unit can sort in one cycle. Must be a power of 2.
# e.g. cmake .. -DSORT_WIDTH=16
if(SORT_WIDTH)
  set(SORT_WIDTH_FLAG "-DSORT_WIDTH=${SORT_WIDTH}")
  message(STATUS "Sort width explicitly set to ${SORT_WIDTH}")
endif()

# Choose the random seed for the hardware compile
# e.g. cmake .. -DSEED=7
if(NOT DEFINED SEED)
    # the default seed
    set(SEED 3)
else()
    message(STATUS "Seed explicitly set to ${SEED}")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED_FLAG "")
else()
    set(SEED_FLAG "-Xsseed=${SEED}")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} -DFPGA_EMULATOR ${BSP_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} ${BSP_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DFPGA_SIMULATOR ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} ${BSP_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} -DFPGA_HARDWARE ${BSP_FLAG}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware ${PROFILE_FLAG} -Xsparallel=2 ${SEED_FLAG} -Xstarget=${FPGA_DEVICE} ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${SORT_WIDTH_FLAG} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

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
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
