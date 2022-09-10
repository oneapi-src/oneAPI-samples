# To see a Makefile equivalent of this build system:
# https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/DPC++/ProjectTemplates/makefile-fpga

set(SOURCE_FILE onchip_memory_cache.cpp)
set(TARGET_NAME onchip_memory_cache)
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
    set(AC_TYPES_FLAG "/Qactypes")
else()
    set(AC_TYPES_FLAG "-qactypes")
endif()

# Allow the user to specify the minimum and maximum cache depth, or select a 
# default value if unspecified
# e.g. cmake .. -DMAX_CACHE_DEPTH=4
if(NOT DEFINED MAX_CACHE_DEPTH)
    set(MAX_CACHE_DEPTH "8")
endif()
if(NOT DEFINED MIN_CACHE_DEPTH)
    set(MIN_CACHE_DEPTH "0")
endif()
set(CACHE_DEPTH_FLAG "-DMAX_CACHE_DEPTH=${MAX_CACHE_DEPTH} -DMIN_CACHE_DEPTH=${MIN_CACHE_DEPTH}")


# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} ${CACHE_DEPTH_FLAG} ${AC_TYPES_FLAG} -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "${AC_TYPES_FLAG} -fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} ${CACHE_DEPTH_FLAG} ${AC_TYPES_FLAG} -fintelfpga")
set(REPORT_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${CACHE_DEPTH_FLAG} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_LINK_FLAGS "${REPORT_LINK_FLAGS} ${AC_TYPES_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR onchip_memory_cache.cpp -o onchip_memory_cache.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o onchip_memory_cache.cpp.o -c onchip_memory_cache.cpp
#    [link]    dpcpp -fintelfpga onchip_memory_cache.cpp.o -o onchip_memory_cache.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early onchip_memory_cache.cpp -o onchip_memory_cache_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> onchip_memory_cache.cpp -o onchip_memory_cache.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o onchip_memory_cache.cpp.o -c onchip_memory_cache.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> onchip_memory_cache.cpp.o -o onchip_memory_cache.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
