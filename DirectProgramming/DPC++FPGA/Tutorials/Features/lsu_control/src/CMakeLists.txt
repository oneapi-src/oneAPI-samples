set(SOURCE_FILE lsu_control.cpp)
set(TARGET_NAME lsu_control)
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
set(EMULATOR_LINK_FLAGS "-fintelfpga")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
IF(SELECTED_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS}")
    set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
    set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
ELSE()
    set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xshyper-optimized-handshaking=off -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS}")
    set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
    set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
ENDIF()

# FPGA emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                       DEPENDS ${SOURCE_FILE})
else()
    add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# FPGA hardware
if(WIN32)
    add_custom_target(fpga COMMAND echo "An FPGA hardware target is not provided on Windows. See README for details.")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# Generate report
if(WIN32)
    set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
    add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})
    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")
    separate_arguments(CMAKE_CXX_FLAGS_LIST WINDOWS_COMMAND "${CMAKE_CXX_FLAGS}")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} COPYONLY)
    add_custom_command(OUTPUT ${DEVICE_OBJ_FILE}
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                       DEPENDS ${SOURCE_FILE})
else()
    set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
    add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_OBJ_FILE}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                       DEPENDS ${SOURCE_FILE})
endif()
