set(DEVICE_SOURCE_FILE kernel.cpp)
set(KERNEL_HEADER_FILE kernel.hpp)
set(HOST_SOURCE_FILE main.cpp)
set(TARGET_NAME fast_recompile)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

String(STRIP "${CMAKE_EXE_LINKER_FLAGS}" CMAKE_EXE_LINKER_FLAGS)

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

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c)

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${_FPGA_BOARD} ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS -fintelfpga -DFPGA_EMULATOR -c)
set(EMULATOR_LINK_FLAGS -fintelfpga)

# fpga emulator
if(WIN32)
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set(HOST_EMU_OBJ "host_emu.o")
    set(DEVICE_EMU_OBJ "dev_emu.o")
    set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

    add_custom_command(OUTPUT ${HOST_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_COMPILE_FLAGS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_EMU_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_EMU_OBJ}
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ} 
                 DEPENDS ${DEVICE_EMU_OBJ})

    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND dpcpp -fintelfpga ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
else()
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set(HOST_EMU_OBJ "host_emu.o")
    set(DEVICE_EMU_OBJ "dev_emu.o")
    set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

    add_custom_command(OUTPUT ${HOST_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_EMU_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_EMU_OBJ}
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ} 
                 DEPENDS ${DEVICE_EMU_OBJ})

    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -fintelfpga ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
endif()

# fpga
if(WIN32)
    add_custom_target(fpga
            COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(HOST_OBJ "host.o")
    set(DEVICE_OBJ "dev.o")
    set(DEVICE_IMAGE_OBJ "dev_image.a")

    add_custom_command(OUTPUT ${HOST_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_OBJ} 
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=image ${DEVICE_OBJ} -o ${DEVICE_IMAGE_OBJ} 
                 DEPENDS ${DEVICE_OBJ})

    add_custom_command(OUTPUT ${FPGA_TARGET} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET} 
                 DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
endif()

# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
