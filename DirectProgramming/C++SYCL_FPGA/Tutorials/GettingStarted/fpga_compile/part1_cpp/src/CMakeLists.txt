
set(SOURCE_FILE vector_add.cpp)
set(TARGET_NAME vector_add)

# FPGA device selection
if(DEFINED FPGA_DEVICE)
    message(STATUS "Ignoring FPGA_DEVICE: ${FPGA_DEVICE}, not applicable")
endif()

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)

add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

# This code sample do not support simulator, report and fpga. 
# following dummy targets are added to be compatiable with reg-tests
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)

add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)

add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})

set(FPGA_TARGET ${TARGET_NAME}.fpga)

add_executable(${FPGA_TARGET} ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})