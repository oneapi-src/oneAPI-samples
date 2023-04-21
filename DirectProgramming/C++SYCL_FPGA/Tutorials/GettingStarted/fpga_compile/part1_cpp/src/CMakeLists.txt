
set(SOURCE_FILE vector_add.cpp)
set(TARGET_NAME vector_add)

# FPGA device selection
if(DEFINED FPGA_DEVICE)
    message(STATUS "Ignoring FPGA_DEVICE: ${FPGA_DEVICE}, not applicable")
endif()

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)

add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
