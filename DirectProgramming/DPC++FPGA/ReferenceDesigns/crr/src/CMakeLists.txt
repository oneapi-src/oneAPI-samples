set(SOURCE_FILE main.cpp)
set(TARGET_NAME crr)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# Intel supported FPGA Boards and their names
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")

# Design specific constant values
set(OUTER_UNROLL_A10 1)
set(INNER_UNROLL_A10 64)
set(OUTER_UNROLL_POW2_A10 1)
set(OUTER_UNROLL_S10 2)
set(INNER_UNROLL_S10 64)
set(OUTER_UNROLL_POW2_S10 2)
set(SEED_A10 1)
set(SEED_S10 4)

# Assume target is the Intel(R) PAC with Intel Arria(R) 10 GX FPGA 
SET(_FPGA_BOARD ${A10_PAC_BOARD_NAME})
SET(OUTER_UNROLL ${OUTER_UNROLL_A10})
SET(INNER_UNROLL ${INNER_UNROLL_A10})
SET(OUTER_UNROLL_POW2 ${OUTER_UNROLL_POW2_A10})
SET(SEED ${SEED_A10})

# Check if target is the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA
IF (NOT DEFINED FPGA_BOARD)
    MESSAGE(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for more information on how to run the design on the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA.")

ELSEIF(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")

ELSEIF(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Stratix(R) 10 SX FPGA.")
    SET(_FPGA_BOARD ${S10_PAC_BOARD_NAME})
    SET(OUTER_UNROLL ${OUTER_UNROLL_S10})
    SET(INNER_UNROLL ${INNER_UNROLL_S10})
    SET(OUTER_UNROLL_POW2 ${OUTER_UNROLL_POW2_S10})
    SET(SEED ${SEED_S10})

ELSE()
    MESSAGE(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
ENDIF()

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${_FPGA_BOARD} ${USER_HARDWARE_FLAGS} -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})
set(FINAL_LINK_FLAGS -fintelfpga -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

#copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)

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

# fpgas
if(WIN32)
    add_custom_target(fpga
                  COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "crr_fpga.o")

    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} ${DEVICE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                   DEPENDS ${DEVICE_FPGA_OBJ})
endif()
                   
# fpga report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    add_custom_command(OUTPUT ${REPORTS_TARGET}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${SOURCE_FILE})

else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CRR_common.hpp CRR_common.hpp COPYONLY)

    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${SOURCE_FILE} CRR_common.hpp)
endif()

# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu data/ordered_inputs.csv -o=data/ordered_output.csv
                  DEPENDS ${TARGET_NAME}.fpga_emu)

