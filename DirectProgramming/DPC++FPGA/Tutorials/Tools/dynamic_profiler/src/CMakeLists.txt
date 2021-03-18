set(SOURCE_FILE dynamic_profiler.cpp)
set(TARGET_NAME dynamic_profiler)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")
set(SELECTED_BOARD ${A10_PAC_BOARD_NAME})
if (NOT DEFINED FPGA_BOARD)
    message(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for information on board selection.")
elseif(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")
elseif(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) D5005 (with Intel Stratix(R) 10 SX FPGA).")
    set(SELECTED_BOARD ${S10_PAC_BOARD_NAME})
else()
    message(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
endif()

# Flags
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -fp-model=precise -ffp-contract=off")
set(EMULATOR_LINK_FLAGS "-fintelfpga -fp-model=precise -ffp-contract=off")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga -fp-model=precise -ffp-contract=off")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${SELECTED_BOARD} -Xsprofile ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation


# FPGA emulator
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})

# FPGA hardware
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})

# Run
add_custom_target(run
                  COMMAND aocl profile -output-dir .. ../${TARGET_NAME}.fpga
                  DEPENDS ${TARGET_NAME}.fpga)
