set(SOURCE_FILE register_map_and_streaming_interfaces.cpp)
set(TARGET_NAME register_map_and_streaming_interfaces)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)

# FPGA device selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Arria10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA device ${FPGA_DEVICE} (Intel(R) Arria(R) 10 FPGAs). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA device ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# This is for finding the install path to the necessary include files
if(DEFINED ENV{INTELFPGAOCLSDKROOT})
   set(SDK_ROOT_PATH $ENV{INTELFPGAOCLSDKROOT})
else()
   message(FATAL_ERROR "The INTELFPGAOCLSDKROOT environment variable must be defined for this tutorial to compile.")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -Wall -Wno-c++2b-extensions ${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR -I${SDK_ROOT_PATH}/include/")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -Wall -Wno-c++2b-extensions ${WIN_FLAG} -fintelfpga -DFPGA_SIMULATOR -I${SDK_ROOT_PATH}/include/")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_SIMULATOR_FLAGS}")

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR register_map_and_streaming_interfaces.cpp -o register_map_and_streaming_interfaces.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o register_map_and_streaming_interfaces.cpp.o -c register_map_and_streaming_interfaces.cpp
#    [link]    icpx -fsycl -fintelfpga register_map_and_streaming_interfaces.cpp.o -o register_map_and_streaming_interfaces.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> register_map_and_streaming_interfaces.cpp -o register_map_and_streaming_interfaces.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o register_map_and_streaming_interfaces.cpp.o -c register_map_and_streaming_interfaces.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> register_map_and_streaming_interfaces.cpp.o -o register_map_and_streaming_interfaces.fpga
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_BOARD> -fsycl-link=early register_map_and_streaming_interfaces.cpp -o register_map_and_streaming_interfaces.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®
