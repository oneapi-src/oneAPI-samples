# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE zero_copy_data_transfer.cpp)
set(TARGET_NAME zero_copy_data_transfer)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# USM is only supported on the PAC S10 board
set(FPGA_BOARD "intel_s10sx_pac:pac_s10_usm")

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -Wall -DFPGA_EMULATOR ${DEVICE_FLAG}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -Wall ${DEVICE_FLAG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xshyper-optimized-handshaking=off -Xsboard=${FPGA_BOARD} ${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR zero_copy_data_transfer.cpp -o zero_copy_data_transfer.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o zero_copy_data_transfer.cpp.o -c zero_copy_data_transfer.cpp
#    [link]    dpcpp -fintelfpga zero_copy_data_transfer.cpp.o -o zero_copy_data_transfer.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE}) # CMake automatically adds #include'd headers to the dependency list
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early zero_copy_data_transfer.cpp -o zero_copy_data_transfer_report.a
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
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> zero_copy_data_transfer.cpp -o zero_copy_data_transfer.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o zero_copy_data_transfer.cpp.o -c zero_copy_data_transfer.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> zero_copy_data_transfer.cpp.o -o zero_copy_data_transfer.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
