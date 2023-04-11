set(SOURCE_FILE zero_copy_data_transfer.cpp)
set(TARGET_NAME zero_copy_data_transfer)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

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

# this tutorial requires USM host allocations. Check the BSP name (which should contain the text 'usm')
# to ensure the BSP has the required support. Allow the user to define USM_HOST_ALLOCATIONS_ENABLED
# to override this check (e.g., cmake .. -DUSM_HOST_ALLOCATIONS_ENABLED=1)
if((IS_BSP STREQUAL "1") AND (NOT FPGA_DEVICE MATCHES ".usm.*") AND (NOT DEFINED USM_HOST_ALLOCATIONS_ENABLED OR USM_HOST_ALLOCATIONS_ENABLED STREQUAL "0"))
    message(FATAL_ERROR "ERROR: This tutorial requires a BSP that has USM host allocations enabled.")
endif()

if (USM_HOST_ALLOCATIONS_ENABLED)
    message(STATUS "USM_HOST_ALLOCATIONS_ENABLED set manually!")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Wall -DFPGA_EMULATOR ${DEVICE_FLAG} ${BSP_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${BSP_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -Wall -DFPGA_SIMULATOR ${DEVICE_FLAG} ${BSP_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xshyper-optimized-handshaking=off -Xstarget=${FPGA_DEVICE} ${DEVICE_FLAG} ${USER_SIMULATOR_FLAGS} ${BSP_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Wall ${DEVICE_FLAG} -DFPGA_HARDWARE ${BSP_FLAG}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xshyper-optimized-handshaking=off -Xstarget=${FPGA_DEVICE} ${DEVICE_FLAG} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR zero_copy_data_transfer.cpp -o zero_copy_data_transfer.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o zero_copy_data_transfer.cpp.o -c zero_copy_data_transfer.cpp
#    [link]    icpx -fsycl -fintelfpga zero_copy_data_transfer.cpp.o -o zero_copy_data_transfer.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE}) # CMake automatically adds #include'd headers to the dependency list
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR zero_copy_data_transfer.cpp -o zero_copy_data_transfer.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o zero_copy_data_transfer.cpp.o -c zero_copy_data_transfer.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation zero_copy_data_transfer.cpp.o -o zero_copy_data_transfer.fpga_emu
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early zero_copy_data_transfer.cpp -o zero_copy_data_transfer_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> zero_copy_data_transfer.cpp -o zero_copy_data_transfer.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o zero_copy_data_transfer.cpp.o -c zero_copy_data_transfer.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> zero_copy_data_transfer.cpp.o -o zero_copy_data_transfer.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
