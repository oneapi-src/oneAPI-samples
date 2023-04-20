set(SOURCE_FILE ac_fixed.cpp)
set(TARGET_NAME ac_fixed)
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

# These are Windows-specific flags:
# 1. /EHsc This is a Windows-specific flag that enables exception handling in host code
# 2. /Qactypes Include ac_types headers and link against ac_types emulation libraries
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes /fp:precise")
else()
    set(AC_TYPES_FLAG "-qactypes -fp-model=precise")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -Wall ${WIN_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_SIMULATOR -Wall ${WIN_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(REPORT_COMPILE_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Wall ${WIN_FLAG} -DFPGA_REPORT")
set(REPORT_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Wall ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early ac_fixed.cpp -o ac_fixed_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${REPORT_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR ${AC_TYPES_FLAG} -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> ac_fixed.cpp -o ac_fixed.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR ${AC_TYPES_FLAG} -o ac_fixed.cpp.o -c ac_fixed.cpp
#   [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> ac_fixed.cpp.o -o ac_fixed.fpga
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> ac_fixed.cpp -o ac_fixed.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -o ac_fixed.cpp.o -c ac_fixed.cpp
#   [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> ac_fixed.cpp.o -o ac_fixed.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
