# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga
set(SOURCE_FILE merge_sort.cpp)
set(TARGET_NAME merge_sort)
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

# check if the BSP has USM host allocations or manually enable using host allocations
# e.g. cmake .. -DUSE_USM_HOST_ALLOCATIONS=1
if(FPGA_BOARD MATCHES ".usm.*" OR DEFINED USE_USM_HOST_ALLOCATIONS)
    set(ENABLE_USM "-DUSM_HOST_ALLOCATIONS")
    message(STATUS "USM host allocations are enabled")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
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

# Choose the random seed for the hardware compile
# e.g. cmake .. -DSEED=7
if(NOT DEFINED SEED)
    # the default seed
    set(SEED 3)
else()
    message(STATUS "Seed explicitly set to ${SEED}")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga ${ENABLE_USM} ${MERGE_UNITS_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${ENABLE_USM} ${MERGE_UNITS_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga ${ENABLE_USM} ${MERGE_UNITS_FLAG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware ${PROFILE_FLAG} -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} ${ENABLE_USM} ${MERGE_UNITS_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR merge_sort.cpp -o merge_sort.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o merge_sort.cpp.o -c merge_sort.cpp
#    [link]    dpcpp -fintelfpga merge_sort.cpp.o -o merge_sort.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=1 -Xsboard=<FPGA_BOARD> -fsycl-link=early merge_sort.cpp -o merge_sort_report.a
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
#   dpcpp -fintelfpga -Xshardware -Xsseed=1 -Xsboard=<FPGA_BOARD> merge_sort.cpp -o merge_sort.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -c -o merge_sort.cpp.o -c merge_sort.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=1 -Xsboard=<FPGA_BOARD> merge_sort.cpp.o -o merge_sort.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.