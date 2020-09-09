set(SOURCE_FILE main.cpp)
set(TARGET_NAME cmake)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)


# Intel supported FPGA cards
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")
set(FPGA_DEVICE_NAME ${A10_PAC_BOARD_NAME})
if (DEFINED FPGA_BOARD)
    set(FPGA_DEVICE_NAME ${FPGA_BOARD})
endif()
# use: cmake -D FPGA_BOARD=<board> to override the default FPGA card


# Compile flags
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR ")
set(EMULATOR_LINK_FLAGS    "-fintelfpga ")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS    "-fintelfpga -Xshardware -Xsboard=${FPGA_DEVICE_NAME} ${USER_HARDWARE_FLAGS}")
# use: cmake -D USER_HARDWARE_FLAGS=<flags> to set additional flags for FPGA backend compilation


# FPGA emulator
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})


# Generate report (without compiling all the way to FPGA hardware)
set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
add_custom_command(OUTPUT ${DEVICE_OBJ_FILE}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                   DEPENDS ${SOURCE_FILE})


# FPGA hardware
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})


# Run FPGA emulator
add_custom_target(run_emu
                  COMMAND ../${TARGET_NAME}.fpga_emu
                  DEPENDS ${TARGET_NAME}.fpga_emu)

# Run FPGA hardware
add_custom_target(run_hw
                  COMMAND ../${TARGET_NAME}.fpga_emu
                  DEPENDS ${TARGET_NAME}.fpga_emu)