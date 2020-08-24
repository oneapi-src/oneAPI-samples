set(SOURCE_FILE fpga_reg.cpp)
set(TARGET_NAME fpga_reg)
set(TARGET_NAME_REG fpga_reg_registered)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_REG ${TARGET_NAME_REG}.fpga)

# Intel supported FPGA Boards and their names
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")

# Assume target is the Intel(R) PAC with Intel Arria(R) 10 GX FPGA 
SET(_FPGA_BOARD ${A10_PAC_BOARD_NAME})

# Check if target is the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA
IF (NOT DEFINED FPGA_BOARD)
    MESSAGE(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for more information on how to run the design on the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA.")

ELSEIF(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")

ELSEIF(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Stratix(R) 10 SX FPGA.")
    SET(_FPGA_BOARD ${S10_PAC_BOARD_NAME})

ELSE()
    MESSAGE(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
ENDIF()

set(HARDWARE_COMPILE_FLAGS "-fintelfpga")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${_FPGA_BOARD} ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

# fpga emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
             COMMAND ${CMAKE_CXX_COMPILER} ${WIN_EMULATOR_COMPILE_FLAGS} /GX ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
             DEPENDS ${SOURCE_FILE})

else()
    add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# fpga
if(WIN32)
    add_custom_target(fpga
            COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_executable(${FPGA_TARGET_REG} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET} ${FPGA_TARGET_REG})

    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})

    set_target_properties(${FPGA_TARGET_REG} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DUSE_FPGA_REG")
    set_target_properties(${FPGA_TARGET_REG} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# report
if(WIN32)
    set(REPORT ${TARGET_NAME}_report.a)
    set(REPORT_REG ${TARGET_NAME_REG}_report.a)

    add_custom_target(report DEPENDS ${REPORT} ${REPORT_REG})

    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME_REG}/${SOURCE_FILE} COPYONLY)

    add_custom_command(OUTPUT ${REPORT}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_REG}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -DUSE_FPGA_REG -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME_REG}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_REG}
                 DEPENDS ${SOURCE_FILE})

else()
    set(REPORT ${TARGET_NAME}_report.a)
    set(REPORT_REG ${TARGET_NAME_REG}_report.a)

    add_custom_target(report DEPENDS ${REPORT} ${REPORT_REG})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${REPORT}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_REG}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -DUSE_FPGA_REG -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_REG}
                 DEPENDS ${SOURCE_FILE})
endif()

# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
