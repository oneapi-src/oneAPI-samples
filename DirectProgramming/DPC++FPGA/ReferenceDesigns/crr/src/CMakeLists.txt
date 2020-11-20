set(SOURCE_FILE main.cpp)
set(TARGET_NAME crr)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

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

# Set design parameters according to the selected board
if(SELECTED_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    # A10 parameters
    set(OUTER_UNROLL 1)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 1)
    set(SEED 1)
elseif(SELECTED_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    # S10 parameters
    set(OUTER_UNROLL 2)
    set(INNER_UNROLL 64)
    set(OUTER_UNROLL_POW2 2)
    set(SEED 4)
else()
    message(FATAL_ERROR "Unknown board!")
endif()

# Flags
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS -fintelfpga -c -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS} -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})
set(FINAL_LINK_FLAGS -fintelfpga -DOUTER_UNROLL=${OUTER_UNROLL} -DINNER_UNROLL=${INNER_UNROLL} -DOUTER_UNROLL_POW2=${OUTER_UNROLL_POW2})

# Copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)


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
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "crr_fpga.o")
    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} ${DEVICE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                       DEPENDS ${DEVICE_FPGA_OBJ})
endif()
                   
# Generate report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )
    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    separate_arguments(CMAKE_CXX_FLAGS_LIST WINDOWS_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS_LIST} ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                       DEPENDS ${SOURCE_FILE})
else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CRR_common.hpp CRR_common.hpp COPYONLY)
    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${SOURCE_FILE} CRR_common.hpp)
endif()

# Run (emulator)
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu data/ordered_inputs.csv -o=data/ordered_output.csv
                  DEPENDS ${TARGET_NAME}.fpga_emu)

