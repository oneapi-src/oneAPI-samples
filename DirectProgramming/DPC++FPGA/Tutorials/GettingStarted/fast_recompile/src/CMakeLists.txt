set(DEVICE_SOURCE_FILE kernel.cpp)
set(KERNEL_HEADER_FILE kernel.hpp)
set(HOST_SOURCE_FILE host.cpp)
set(TARGET_NAME fast_recompile)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

set(EMULATOR_COMPILE_FLAGS ${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR -c)
set(EMULATOR_LINK_FLAGS -fintelfpga)
set(HARDWARE_COMPILE_FLAGS ${WIN_FLAG} -fintelfpga -c)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS})
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -DFPGA_EMULATOR -c host.cpp -o host_emu.o
#   dpcpp -fintelfpga -DFPGA_EMULATOR -c kernel.cpp -o dev_emu.o
#   dpcpp -fintelfpga -fsycl-link=image dev_emu.o -o dev_image_emu.a
#   dpcpp -fintelfpga host_emu.o dev_image_emu.a -o fast_recompile.fpga

if(WIN32)
    set(EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
endif()
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set(HOST_EMU_OBJ "host_emu.o")
set(DEVICE_EMU_OBJ "dev_emu.o")
set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

set(CMAKE_CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS_LIST})
separate_arguments(CMAKE_CXX_FLAGS_LIST)

add_custom_command(OUTPUT ${HOST_EMU_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${EMULATOR_COMPILE_FLAGS}
                   ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_EMU_OBJ}
                   DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${DEVICE_EMU_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_EMU_OBJ}
                   DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ}
                   DEPENDS ${DEVICE_EMU_OBJ})

add_custom_command(OUTPUT ${EMULATOR_TARGET}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} -fintelfpga ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET}
                   DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})

###############################################################################
### FPGA Hardware
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -c host.cpp -o host.o
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=image kernel.cpp -o dev_image.a
#   dpcpp -fintelfpga host.o dev_image.a -o fast_recompile.fpga

if(WIN32)
    set(FPGA_TARGET ${FPGA_TARGET}.exe)
endif()
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set(HOST_OBJ "host.o")
set(DEVICE_OBJ "dev.o")
set(DEVICE_IMAGE_OBJ "dev_image.a")

set(CMAKE_CXX_FLAGS_LIST "${CMAKE_CXX_FLAGS}")
separate_arguments(CMAKE_CXX_FLAGS_LIST)
set(HARDWARE_LINK_FLAGS_LIST "${HARDWARE_LINK_FLAGS}")
separate_arguments(HARDWARE_LINK_FLAGS_LIST)

add_custom_command(OUTPUT ${HOST_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_OBJ}
                   DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=image ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_IMAGE_OBJ}
                   DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${FPGA_TARGET}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                   DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
