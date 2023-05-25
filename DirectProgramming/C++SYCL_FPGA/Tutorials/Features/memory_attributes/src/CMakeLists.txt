set(SOURCE_FILE memory_attributes.cpp)
set(TARGET_NAME memory_attributes)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1")
        set(IS_BSP "1")
    else()
        set(IS_BSP "0")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number.")
        message(STATUS "If the target is actually a BSP, run cmake with -DIS_BSP=1.")
    endif()
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

if ((IS_BSP STREQUAL "1"))
    set(CLOCK_FLAG "")
else()
    set(CLOCK_FLAG "-Xsuse-2xclock")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DFPGA_SIMULATOR ${CLOCK_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${CLOCK_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_HARDWARE ${CLOCK_FLAG}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${CLOCK_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR memory_attributes.cpp -o memory_attributes.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o memory_attributes.cpp.o -c memory_attributes.cpp
#    [link]    icpx -fsycl -fintelfpga memory_attributes.cpp.o -o memory_attributes.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR mem_channel.cpp -o mem_channel.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o mem_channel.cpp.o -c mem_channel.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation  mem_channel.cpp.o -o mem_channel.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early memory_attributes.cpp -o memory_attributes_report.a
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
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> memory_attributes.cpp -o memory_attributes.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o memory_attributes.cpp.o -c memory_attributes.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> memory_attributes.cpp.o -o memory_attributes.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
