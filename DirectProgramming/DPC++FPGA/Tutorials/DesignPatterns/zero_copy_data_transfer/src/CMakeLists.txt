set(SOURCE_FILE zero_copy_data_transfer.cpp)
set(TARGET_NAME zero_copy_data_transfer)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# USM is only supported on the PACS10 board
set(S10_PAC_USM_BOARD_NAME "intel_s10sx_pac:pac_s10_usm")

set(HARDWARE_COMPILE_FLAGS "-fintelfpga -c")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Wall -Xshardware -Xsboard=${S10_PAC_USM_BOARD_NAME} ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -Wall -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

# fpga emulator
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# fpga
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# fpga report
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/buffer_kernel.hpp buffer_kernel.hpp COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/restricted_usm_kernel.hpp restricted_usm_kernel.hpp COPYONLY)

separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")

add_custom_command(OUTPUT ${REPORTS_TARGET}
             COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
             DEPENDS ${SOURCE_FILE})
add_custom_target(report DEPENDS ${REPORTS_TARGET})
endif()

