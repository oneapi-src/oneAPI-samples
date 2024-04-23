set(REGISTER_MAP_FUNCTOR_SOURCE_FILE reg_map_functor.cpp)
set(REGISTER_MAP_FUNCTOR_TARGET_NAME reg_map_functor)
set(REGISTER_MAP_FUNCTOR_EMULATOR_TARGET ${REGISTER_MAP_FUNCTOR_TARGET_NAME}.fpga_emu)
set(REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET ${REGISTER_MAP_FUNCTOR_TARGET_NAME}.fpga_sim)
set(REGISTER_MAP_FUNCTOR_FPGA_TARGET ${REGISTER_MAP_FUNCTOR_TARGET_NAME}.fpga)

set(STREAMING_FUNCTOR_SOURCE_FILE stream_functor.cpp)
set(STREAMING_FUNCTOR_TARGET_NAME stream_functor)
set(STREAMING_FUNCTOR_EMULATOR_TARGET ${STREAMING_FUNCTOR_TARGET_NAME}.fpga_emu)
set(STREAMING_FUNCTOR_SIMULATOR_TARGET ${STREAMING_FUNCTOR_TARGET_NAME}.fpga_sim)
set(STREAMING_FUNCTOR_FPGA_TARGET ${STREAMING_FUNCTOR_TARGET_NAME}.fpga)

set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SOURCE_FILE stream_rm_stall.cpp)
set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_TARGET_NAME stream_rm_stall)
set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_TARGET_NAME}.fpga_emu)
set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_TARGET_NAME}.fpga_sim)
set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_TARGET_NAME}.fpga)

set(REGISTER_MAP_LAMBDA_SOURCE_FILE reg_map_lambda.cpp)
set(REGISTER_MAP_LAMBDA_TARGET_NAME reg_map_lambda)
set(REGISTER_MAP_LAMBDA_EMULATOR_TARGET ${REGISTER_MAP_LAMBDA_TARGET_NAME}.fpga_emu)
set(REGISTER_MAP_LAMBDA_SIMULATOR_TARGET ${REGISTER_MAP_LAMBDA_TARGET_NAME}.fpga_sim)
set(REGISTER_MAP_LAMBDA_FPGA_TARGET ${REGISTER_MAP_LAMBDA_TARGET_NAME}.fpga)

set(STREAMING_LAMBDA_SOURCE_FILE stream_lambda.cpp)
set(STREAMING_LAMBDA_TARGET_NAME stream_lambda)
set(STREAMING_LAMBDA_EMULATOR_TARGET ${STREAMING_LAMBDA_TARGET_NAME}.fpga_emu)
set(STREAMING_LAMBDA_SIMULATOR_TARGET ${STREAMING_LAMBDA_TARGET_NAME}.fpga_sim)
set(STREAMING_LAMBDA_FPGA_TARGET ${STREAMING_LAMBDA_TARGET_NAME}.fpga)

set(STREAMING_PIPELINED_FUNCTOR_SOURCE_FILE stream_pipelined.cpp)
set(STREAMING_PIPELINED_FUNCTOR_TARGET_NAME stream_pipelined)
set(STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET ${STREAMING_PIPELINED_FUNCTOR_TARGET_NAME}.fpga_emu)
set(STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET ${STREAMING_PIPELINED_FUNCTOR_TARGET_NAME}.fpga_sim)
set(STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET ${STREAMING_PIPELINED_FUNCTOR_TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA family: ${FPGA_DEVICE} \
                    \nPlease refer to the README for information on target selection.")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1")
        set(IS_BSP "1")
    else()
        set(IS_BSP "0")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number.")
        message(STATUS "If the target is actually a BSP, run cmake with -DIS_BSP=1.")
    endif()
endif()

if((IS_BSP STREQUAL "1"))
    message(FATAL_ERROR "ERROR: This tutorial is only supported in the SYCL HLS flow and therefore does not support BSPs as a target.")
endif()


# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
include_directories(${SDK_ROOT_PATH}/include/)
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall -Wno-c++2b-extensions ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR reg_map_functor.cpp -o reg_map_functor.fpga_emu
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR stream_functor.cpp -o stream_functor.fpga_emu
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR stream_rm_stall.cpp -o stream_rm_stall.fpga_emu
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR reg_map_lambda.cpp -o reg_map_lambda.fpga_emu
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR stream_lambda.cpp -o stream_lambda.fpga_emu
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR stream_pipelined.cpp -o stream_pipelined.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o reg_map_functor.cpp.o -c reg_map_functor.cpp
#    [link]    icpx -fsycl -fintelfpga reg_map_functor.cpp.o -o reg_map_functor.fpga_emu
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o stream_functor.cpp.o -c stream_functor.cpp
#    [link]    icpx -fsycl -fintelfpga stream_functor.cpp.o -o stream_functor.fpga_emu
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o stream_rm_stall.cpp.o -c stream_rm_stall.cpp
#    [link]    icpx -fsycl -fintelfpga stream_rm_stall.cpp.o -o stream_rm_stall.fpga_emu
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o reg_map_lambda.cpp.o -c reg_map_lambda.cpp
#    [link]    icpx -fsycl -fintelfpga reg_map_lambda.cpp.o -o reg_map_lambda.fpga_emu
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o stream_lambda.cpp.o -c stream_lambda.cpp
#    [link]    icpx -fsycl -fintelfpga stream_lambda.cpp.o -o stream_lambda.fpga_emu
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o stream_pipelined.cpp.o -c stream_pipelined.cpp
#    [link]    icpx -fsycl -fintelfpga stream_pipelined.cpp.o -o stream_pipelined.fpga_emu
add_executable(${REGISTER_MAP_FUNCTOR_EMULATOR_TARGET} ${REGISTER_MAP_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_FUNCTOR_EMULATOR_TARGET} ${STREAMING_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${REGISTER_MAP_LAMBDA_EMULATOR_TARGET} ${REGISTER_MAP_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_LAMBDA_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_LAMBDA_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_LAMBDA_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_LAMBDA_EMULATOR_TARGET} ${STREAMING_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_LAMBDA_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_LAMBDA_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_LAMBDA_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET} ${STREAMING_PIPELINED_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_custom_target(fpga_emu DEPENDS ${REGISTER_MAP_FUNCTOR_EMULATOR_TARGET} ${STREAMING_FUNCTOR_EMULATOR_TARGET} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_EMULATOR_TARGET} ${REGISTER_MAP_LAMBDA_EMULATOR_TARGET} ${STREAMING_LAMBDA_EMULATOR_TARGET} ${STREAMING_PIPELINED_FUNCTOR_EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_functor.cpp -o reg_map_functor.fpga_sim
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_functor.cpp -o stream_functor.fpga_sim
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_rm_stall.cpp -o stream_rm_stall.fpga_sim
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_lambda.cpp -o reg_map_lambda.fpga_sim
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_lambda.cpp -o stream_lambda.fpga_sim
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_pipelined.cpp -o stream_pipelined.fpga_sim
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o reg_map_functor.cpp.o -c reg_map_functor.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_functor.cpp.o -o reg_map_functor.fpga_sim
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o stream_functor.cpp.o -c stream_functor.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_functor.cpp.o -o stream_functor.fpga_sim
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o stream_rm_stall.cpp.o -c stream_rm_stall.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_rm_stall.cpp.o -o stream_rm_stall.fpga_sim
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o reg_map_lambda.cpp.o -c reg_map_lambda.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_lambda.cpp.o -o reg_map_lambda.fpga_sim
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o stream_lambda.cpp.o -c stream_lambda.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_lambda.cpp.o -o stream_lambda.fpga_sim
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o stream_pipelined.cpp.o -c stream_pipelined.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_pipelined.cpp.o -o stream_pipelined.fpga_sim
add_executable(${REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET} ${REGISTER_MAP_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${REGISTER_MAP_LAMBDA_SIMULATOR_TARGET} ${REGISTER_MAP_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_LAMBDA_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_LAMBDA_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_LAMBDA_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_LAMBDA_SIMULATOR_TARGET} ${STREAMING_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_LAMBDA_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_LAMBDA_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_LAMBDA_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_executable(${STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_PIPELINED_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

add_custom_target(fpga_sim DEPENDS ${REGISTER_MAP_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_FUNCTOR_SIMULATOR_TARGET} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SIMULATOR_TARGET} ${REGISTER_MAP_LAMBDA_SIMULATOR_TARGET} ${STREAMING_LAMBDA_SIMULATOR_TARGET} ${STREAMING_PIPELINED_FUNCTOR_SIMULATOR_TARGET})

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_functor.cpp -o reg_map_functor.fpga
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_functor.cpp -o stream_functor.fpga
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_rm_stall.cpp -o stream_rm_stall.fpga
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_lambda.cpp -o reg_map_lambda.fpga
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_lambda.cpp -o stream_lambda.fpga
#   icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_pipelined.cpp -o stream_pipelined.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o reg_map_functor.cpp.o -c reg_map_functor.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_functor.cpp.o -o reg_map_functor.fpga
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o stream_functor.cpp.o -c stream_functor.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_functor.cpp.o -o stream_functor.fpga
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o stream_rm_stall.cpp.o -c stream_rm_stall.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_rm_stall.cpp.o -o stream_rm_stall.fpga
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o reg_map_lambda.cpp.o -c reg_map_lambda.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> reg_map_lambda.cpp.o -o reg_map_lambda.fpga
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o stream_lambda.cpp.o -c stream_lambda.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware  -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_lambda.cpp.o -o stream_lambda.fpga
#   [compile] icpx -fsycl -fintelfpga -DFPGA_HARDWARE -o stream_pipelined.cpp.o -c stream_pipelined.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware  -Xstarget=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> stream_pipelined.cpp.o -o stream_pipelined.fpga
add_executable(${REGISTER_MAP_FUNCTOR_FPGA_TARGET} ${REGISTER_MAP_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_FUNCTOR_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_FUNCTOR_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_FUNCTOR_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${REGISTER_MAP_FUNCTOR_FPGA_TARGET}")

add_executable(${STREAMING_FUNCTOR_FPGA_TARGET} ${STREAMING_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_FUNCTOR_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_FUNCTOR_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_FUNCTOR_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${STREAMING_FUNCTOR_FPGA_TARGET}")

add_executable(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET}")

add_executable(${REGISTER_MAP_LAMBDA_FPGA_TARGET} ${REGISTER_MAP_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_LAMBDA_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_LAMBDA_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_LAMBDA_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${REGISTER_MAP_LAMBDA_FPGA_TARGET}")

add_executable(${STREAMING_LAMBDA_FPGA_TARGET} ${STREAMING_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_LAMBDA_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_LAMBDA_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_LAMBDA_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${STREAMING_LAMBDA_FPGA_TARGET}")

add_executable(${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET} ${STREAMING_PIPELINED_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET}")

add_custom_target(fpga DEPENDS ${REGISTER_MAP_FUNCTOR_FPGA_TARGET} ${STREAMING_FUNCTOR_FPGA_TARGET} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_TARGET} ${REGISTER_MAP_LAMBDA_FPGA_TARGET} ${STREAMING_LAMBDA_FPGA_TARGET} ${STREAMING_PIPELINED_FUNCTOR_FPGA_TARGET})
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early reg_map_functor.cpp -o reg_map_functor_report.a
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early stream_functor.cpp -o stream_functor_report.a
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early stream_rm_stall.cpp -o stream_rm_stall_report.a
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early reg_map_lambda.cpp -o reg_map_lambda_report.a
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early stream_lambda.cpp -o stream_lambda_report.a
#   icpx -fsycl -fintelfpga -Xssimulation -Xsboard=<FPGA_FAMILY/FPGA_DEVICE_PART_NUMBER> -fsycl-link=early stream_pipelined.cpp -o stream_pipelined_report.a
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
set(REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE ${REGISTER_MAP_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE} ${REGISTER_MAP_FUNCTOR_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_FUNCTOR_FPGA_EARLY_IMAGE ${STREAMING_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${STREAMING_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set(REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE ${REGISTER_MAP_LAMBDA_TARGET_NAME}_report.a)
add_executable(${REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE} ${REGISTER_MAP_LAMBDA_SOURCE_FILE})
target_include_directories(${REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_LAMBDA_FPGA_EARLY_IMAGE ${STREAMING_LAMBDA_TARGET_NAME}_report.a)
add_executable(${STREAMING_LAMBDA_FPGA_EARLY_IMAGE} ${STREAMING_LAMBDA_SOURCE_FILE})
target_include_directories(${STREAMING_LAMBDA_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_LAMBDA_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set(STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE ${STREAMING_PIPELINED_FUNCTOR_TARGET_NAME}_report.a)
add_executable(${STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_PIPELINED_FUNCTOR_SOURCE_FILE})
target_include_directories(${STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE} PRIVATE ../../../../../include)
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

add_custom_target(report DEPENDS ${REGISTER_MAP_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_FUNCTOR_FPGA_EARLY_IMAGE} ${STREAMING_RM_DOWNSTREAM_STALL_FUNCTOR_FPGA_EARLY_IMAGE} ${REGISTER_MAP_LAMBDA_FPGA_EARLY_IMAGE} ${STREAMING_LAMBDA_FPGA_EARLY_IMAGE} ${STREAMING_PIPELINED_FUNCTOR_FPGA_EARLY_IMAGE})
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®
