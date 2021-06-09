set(DEVICE_SOURCE_FILE qrd.cpp)
set(DEVICE_HEADER_FILE qrd.hpp)
set(HOST_SOURCE_FILE qrd_demo.cpp)
set(TARGET_NAME qrd)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# Intel supported FPGA Boards and their names
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")

# Design specific constant values
set(ROWS_COMPONENT_A10 128)
set(COLS_COMPONENT_A10 128)

set(ROWS_COMPONENT_S10 256)
set(COLS_COMPONENT_S10 256)

set(FIXED_ITERATIONS_A10 64)
set(FIXED_ITERATIONS_S10 105)

set(SEED_A10 9)
set(SEED_S10 5)

set(CLOCK_TARGET_A10 360MHz)
set(CLOCK_TARGET_S10 480MHz)

# Set parameter values assuming target is Intel(R) PAC with Intel Arria(R) 10 GX FPGA
set(SELECTED_BOARD ${A10_PAC_BOARD_NAME})
set(FIXED_ITERATIONS ${FIXED_ITERATIONS_A10})
set(CLOCK_TARGET ${CLOCK_TARGET_A10})
set(SEED ${SEED_A10})
set(ROWS_COMPONENT ${ROWS_COMPONENT_A10})
set(COLS_COMPONENT ${COLS_COMPONENT_A10})

# Check if target is the Intel(R) PAC D5005 (with Intel Stratix(R) 10 SX FPGA)
if(NOT DEFINED FPGA_BOARD)
    message(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for information on board selection.")
elseif(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")
elseif(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) D5005 (with Intel Stratix(R) 10 SX FPGA).")
    set(SELECTED_BOARD ${S10_PAC_BOARD_NAME})
    set(FIXED_ITERATIONS ${FIXED_ITERATIONS_S10})
    set(CLOCK_TARGET ${CLOCK_TARGET_S10})
    set(SEED ${SEED_S10})
    set(ROWS_COMPONENT ${ROWS_COMPONENT_S10})
    set(COLS_COMPONENT ${COLS_COMPONENT_S10})
else()
    message(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
endif()

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c -fno-fast-math -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsclock=${CLOCK_TARGET} -Xsfp-relaxed -fno-fast-math -Xsparallel=2 -Xsseed=${SEED} -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS} -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})
set(FINAL_LINK_FLAGS -fintelfpga -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT})

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(EMULATOR_LINK_FLAGS -fintelfpga )

# FPGA emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${WIN_EMULATOR_COMPILE_FLAGS} /F 2097152 /EHsc ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
else()
    add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# FPGA hardware
if(WIN32)
    add_custom_target(fpga COMMAND echo "An FPGA hardware target is not provided on Windows. See README for details.")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "qrd_fpga.o")
    set(DEVICE_IMAGE_FPGA_OBJ "qrd_fpga.a")
    set(HOST_FPGA_OBJ "qrd_host.o")

    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")

    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})

    add_custom_command(OUTPUT ${HOST_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_FPGA_OBJ}
                       DEPENDS ${HOST_SOURCE_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} -fsycl-link=image ${DEVICE_FPGA_OBJ} -o ${DEVICE_IMAGE_FPGA_OBJ}
                       DEPENDS ${DEVICE_FPGA_OBJ})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${FINAL_LINK_FLAGS} ${HOST_FPGA_OBJ} ${DEVICE_IMAGE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                       DEPENDS ${DEVICE_IMAGE_FPGA_OBJ} ${HOST_FPGA_OBJ})
endif()

# FPGA hardware report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )
    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    separate_arguments(CMAKE_CXX_FLAGS_LIST WINDOWS_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${REPORTS_TARGET}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS_LIST} ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})
else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${DEVICE_SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_HEADER_FILE} ${DEVICE_HEADER_FILE} COPYONLY)

    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} -fsycl-link ${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})
endif()

# Run emulator
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu
                  DEPENDS ${TARGET_NAME}.fpga_emu)

