set(REGISTER_MAP_CONTROL_FUNCTOR_SOURCE_FILE register_map_control_functor_model.cpp)
set(REGISTER_MAP_CONTROL_FUNCTOR_TARGET_NAME register_map_control_functor_model)
set(REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET ${REGISTER_MAP_CONTROL_FUNCTOR_TARGET_NAME}.fpga_emu)
set(REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET ${REGISTER_MAP_CONTROL_FUNCTOR_TARGET_NAME}.fpga_sim)

set(STREAMING_CONTROL_FUNCTOR_SOURCE_FILE streaming_control_functor_model.cpp)
set(STREAMING_CONTROL_FUNCTOR_TARGET_NAME streaming_control_functor_model)
set(STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET ${STREAMING_CONTROL_FUNCTOR_TARGET_NAME}.fpga_emu)
set(STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET ${STREAMING_CONTROL_FUNCTOR_TARGET_NAME}.fpga_sim)

set(REGISTER_MAP_CONTROL_LAMBDA_SOURCE_FILE register_map_control_lambda_model.cpp)
set(REGISTER_MAP_CONTROL_LAMBDA_TARGET_NAME register_map_control_lambda_model)
set(REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET ${REGISTER_MAP_CONTROL_LAMBDA_TARGET_NAME}.fpga_emu)
set(REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET ${REGISTER_MAP_CONTROL_LAMBDA_TARGET_NAME}.fpga_sim)

set(STREAMING_CONTROL_LAMBDA_SOURCE_FILE streaming_control_lambda_model.cpp)
set(STREAMING_CONTROL_LAMBDA_TARGET_NAME streaming_control_lambda_model)
set(STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET ${STREAMING_CONTROL_LAMBDA_TARGET_NAME}.fpga_emu)
set(STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET ${STREAMING_CONTROL_LAMBDA_TARGET_NAME}.fpga_sim)

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
add_executable(${REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET} ${REGISTER_MAP_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET} ${STREAMING_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET} ${REGISTER_MAP_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET} ${STREAMING_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_custom_target(fpga_emu DEPENDS ${REGISTER_MAP_CONTROL_FUNCTOR_EMULATOR_TARGET} ${STREAMING_CONTROL_FUNCTOR_EMULATOR_TARGET} ${REGISTER_MAP_CONTROL_LAMBDA_EMULATOR_TARGET} ${STREAMING_CONTROL_LAMBDA_EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> register_map_and_streaming_interfaces.cpp -o register_map_and_streaming_interfaces.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o register_map_and_streaming_interfaces.cpp.o -c register_map_and_streaming_interfaces.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> register_map_and_streaming_interfaces.cpp.o -o register_map_and_streaming_interfaces.fpga
add_executable(${REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET} ${REGISTER_MAP_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET} ${REGISTER_MAP_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET} ${STREAMING_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_custom_target(fpga_sim DEPENDS ${REGISTER_MAP_CONTROL_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_CONTROL_FUNCTOR_SIMULATOR_TARGET} ${REGISTER_MAP_CONTROL_LAMBDA_SIMULATOR_TARGET} ${STREAMING_CONTROL_LAMBDA_SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_BOARD> -fsycl-link=early register_map_and_streaming_interfaces.cpp -o register_map_and_streaming_interfaces.a
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
set(REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE ${REGISTER_MAP_CONTROL_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} ${REGISTER_MAP_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE ${STREAMING_CONTROL_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_CONTROL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -fsycl-link=early")

set(REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE ${REGISTER_MAP_CONTROL_LAMBDA_TARGET_NAME}_report.a)
add_executable(${REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} ${REGISTER_MAP_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE ${STREAMING_CONTROL_LAMBDA_TARGET_NAME}_report.a)
add_executable(${STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} ${STREAMING_CONTROL_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -fsycl-link=early")

add_custom_target(report DEPENDS ${REGISTER_MAP_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_CONTROL_FUNCTOR_FPGA_EARLY_IMAGE} ${REGISTER_MAP_CONTROL_LAMBDA_FPGA_EARLY_IMAGE} ${STREAMING_CONTROL_LAMBDA_FPGA_EARLY_IMAGE})
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®
