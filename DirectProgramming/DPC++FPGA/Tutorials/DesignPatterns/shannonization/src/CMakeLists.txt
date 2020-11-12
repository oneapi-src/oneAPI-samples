set(SOURCE_FILE shannonization.cpp)
set(TARGET_NAME shannonization)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

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

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
IF(_FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${_FPGA_BOARD} -DA10 ${USER_HARDWARE_FLAGS}")
    set(HARDWARE_COMPILE_FLAGS "-fintelfpga -DA10")
    set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -DA10")
ELSE()
    set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xshyper-optimized-handshaking=off -Xsboard=${_FPGA_BOARD} -DS10 ${USER_HARDWARE_FLAGS}")
    set(HARDWARE_COMPILE_FLAGS "-fintelfpga -DS10")
    set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -DS10")
ENDIF()

set(EMULATOR_LINK_FLAGS "-fintelfpga")

# fpga emulator
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(fpga_emu_v0 DEPENDS ${EMULATOR_TARGET})

    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
endif()

# fpga
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# report
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/IntersectionKernel.hpp IntersectionKernel.hpp COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                 DEPENDS ${SOURCE_FILE} IntersectionKernel.hpp)
    add_custom_target(report DEPENDS ${REPORTS_TARGET})

endif()

# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
