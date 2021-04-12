# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE fp_pragmas.cpp)
set(TARGET_NAME fp_pragmas)
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
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation


###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR fp_pragmas.cpp -o fp_pragmas.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o fp_pragmas.cpp.o -c fp_pragmas.cpp
#    [link]    dpcpp -fintelfpga fp_pragmas.cpp.o -o fp_pragmas.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early fp_pragmas.cpp -o fp_pragmas_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set(CMAKE_CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
separate_arguments(CMAKE_CXX_FLAGS_LIST)
set(HARDWARE_LINK_FLAGS_LIST ${HARDWARE_LINK_FLAGS})
separate_arguments(HARDWARE_LINK_FLAGS_LIST)
add_custom_command(OUTPUT ${FPGA_EARLY_IMAGE}
                   COMMAND ${CMAKE_CXX_COMPILER} ${WIN_FLAG} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=early ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${FPGA_EARLY_IMAGE}
                   DEPENDS ${SOURCE_FILE})
                   # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> fp_pragmas.cpp -o fp_pragmas.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fp_pragmas.cpp.o -c fp_pragmas.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> fp_pragmas.cpp.o -o fp_pragmas.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")

# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
