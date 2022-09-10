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

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR host.cpp kernel.cpp -o fast_recompile.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o host.cpp.o -c host.cpp
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o kernel.cpp.o -c kernel.cpp
#    [link]    dpcpp -fintelfpga host.cpp.o kernel.cpp.o -o fast_recompile.fpga_emu
add_executable(${EMULATOR_TARGET} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early host.cpp kernel.cpp -o fast_compile_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®


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
set(HARDWARE_COMPILE_FLAGS_LIST "${HARDWARE_COMPILE_FLAGS}")
separate_arguments(HARDWARE_COMPILE_FLAGS_LIST)
set(HARDWARE_LINK_FLAGS_LIST "${HARDWARE_LINK_FLAGS}")
separate_arguments(HARDWARE_LINK_FLAGS_LIST)
 

add_custom_command(OUTPUT ${HOST_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS_LIST} -c ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_OBJ}
                   DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=image ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_IMAGE_OBJ}
                   DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${FPGA_TARGET}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                   DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
