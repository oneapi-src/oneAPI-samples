# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE loop_initiation_interval.cpp)
set(TARGET_NAME loop_ii)
set(TARGET_NAME_ENABLE_II loop_ii_enable_ii)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_ENABLE_II ${TARGET_NAME_ENABLE_II}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    set(DEVICE_FLAG "-DA10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    if(FPGA_BOARD STREQUAL "intel_a10gx_pac:pac_a10")
        set(DEVICE_FLAG "-DA10")
    elseif(FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10" OR FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10_usm")
        set(DEVICE_FLAG "-DS10")
    endif()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Please make sure you have set -DDEVICE_FLAG=-DA10, -DDEVICE_FLAG=-DS10 or \
                         -DDEVICE_FLAG=-DAgilex.")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR ${DEVICE_FLAG}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga ${DEVICE_FLAG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR -DA10 fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -DA10 -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    dpcpp -fintelfpga fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -DA10 -fsycl-link=early loop_initiation_interval.cpp -o loop_ii_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
set(FPGA_EARLY_IMAGE_ENABLE_II ${TARGET_NAME_ENABLE_II}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
add_executable(${FPGA_EARLY_IMAGE_ENABLE_II} ${SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} ${FPGA_EARLY_IMAGE_ENABLE_II})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
set_target_properties(${FPGA_EARLY_IMAGE_ENABLE_II} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DENABLE_II")
set_target_properties(${FPGA_EARLY_IMAGE_ENABLE_II} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -DA10 loop_initiation_interval.cpp -o loop_ii.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o loop_initiation_interval.cpp.o -DA10 -c loop_initiation_interval.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -DA10 loop_initiation_interval.cpp.o -o loop_ii.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_executable(${FPGA_TARGET_ENABLE_II} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET} ${FPGA_TARGET_ENABLE_II})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
set_target_properties(${FPGA_TARGET_ENABLE_II} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DENABLE_II")
set_target_properties(${FPGA_TARGET_ENABLE_II} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_ENABLE_II}")

