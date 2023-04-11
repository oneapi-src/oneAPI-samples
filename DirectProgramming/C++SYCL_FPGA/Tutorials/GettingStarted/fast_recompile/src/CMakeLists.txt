set(DEVICE_SOURCE_FILE kernel.cpp)
set(KERNEL_HEADER_FILE kernel.hpp)
set(HOST_SOURCE_FILE host.cpp)
set(TARGET_NAME fast_recompile)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR host.cpp kernel.cpp -o fast_recompile.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o host.cpp.o -c host.cpp
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o kernel.cpp.o -c kernel.cpp
#    [link]    icpx -fsycl -fintelfpga host.cpp.o kernel.cpp.o -o fast_recompile.fpga_emu
add_executable(${EMULATOR_TARGET} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR <file>.cpp -o <file>.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> <file>.cpp.o -o <file>.fpga_sim
add_executable(${SIMULATOR_TARGET} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early host.cpp kernel.cpp -o fast_compile_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®


###############################################################################
### FPGA Hardware
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -c host.cpp -o host.o
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=image kernel.cpp -o dev_image.a
#   icpx -fsycl -fintelfpga host.o dev_image.a -o fast_recompile.fpga

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
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS_LIST} -c ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_OBJ} -I${CMAKE_CURRENT_LIST_DIR}/../../../../include
                   DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=image ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_IMAGE_OBJ} -I${CMAKE_CURRENT_LIST_DIR}/../../../../include
                   DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

add_custom_command(OUTPUT ${FPGA_TARGET}
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} -fsycl -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET} -I${CMAKE_CURRENT_LIST_DIR}/../../../../include
                   DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
