###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILE matmul_demo.cpp)
set(TARGET_NAME matmul)

set(ROWS_A 8)
set(COMMON 64)
set(COLS_B 8)
set(TILE_A 8)
set(TILE_B 8)
set(TILE_COMMON 64)
set(DEBUG 0)

if(DEFINED SET_ROWS_A)
    set(ROWS_A ${SET_ROWS_A})
endif()
if(DEFINED SET_COMMON)
    set(COMMON ${SET_COMMON})
endif()
if(DEFINED SET_COLS_B)
    set(COLS_B ${SET_COLS_B})
endif()
if(DEFINED SET_TILE_A)
    set(TILE_A ${SET_TILE_A})
endif()
if(DEFINED SET_TILE_B)
    set(TILE_B ${SET_TILE_B})
endif()
if(DEFINED SET_TILE_COMMON)
    set(TILE_COMMON ${SET_TILE_COMMON})
endif()
if(DEFINED SET_DEBUG)
    set(DEBUG ${SET_DEBUG})
endif()

message(STATUS "ROWS_A=${ROWS_A}")
message(STATUS "COMMON=${COMMON}")
message(STATUS "COLS_B=${COLS_B}")
message(STATUS "TILE_A=${TILE_A}")
message(STATUS "TILE_B=${TILE_B}")
message(STATUS "TILE_COMMON=${TILE_COMMON}")

set(MATMUL_PARAMS "-DROWS_A=${ROWS_A} -DCOMMON=${COMMON} -DCOLS_B=${COLS_B} -DTILE_A=${TILE_A} -DTILE_B=${TILE_B} -DTILE_COMMON=${TILE_COMMON} -DDEBUG=${DEBUG}")

# Use cmake -DFPGA_DEVICE=<board-support-package>:<board-variant> to choose a
# different device. Here are a few device examples (this list is not
# exhaustive):
#   intel_s10sx_pac:pac_s10
#   intel_s10sx_pac:pac_s10_usm
#   intel_a10gx_pac:pac_a10
# Note that depending on your installation, you may need to specify the full 
# path to the board support package (BSP), this usually is in your install 
# folder.
#
# You can also specify a device family (E.g. "Arria10" or "Stratix10") or a
# specific part number (E.g. "10AS066N3F40E2SG") to generate a standalone IP.
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
endif()

# Use cmake -DUSER_FPGA_FLAGS=<flags> to set extra flags for FPGA backend
# compilation. 
set(USER_FPGA_FLAGS "${USER_FPGA_FLAGS}")

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS "${USER_FLAGS} -Xsseed=7 -Xsclock=360MHz ${MATMUL_PARAMS}")
###############################################################################
### no changes after here
###############################################################################

set(EMULATOR_EXTENSION fpga_emu)
set(REPORT_EXTENSION report)
set(SIMULATOR_EXTENSION fpga_sim)
set(FPGA_EXTENSION fpga)
set(IP_EXPORT_EXTENSION fpga_ip_export)

# define the different build targets the generated build script will support.
set(EMULATOR_TARGET ${TARGET_NAME}.${EMULATOR_EXTENSION})
set(REPORT_TARGET ${TARGET_NAME}.${REPORT_EXTENSION})
set(SIMULATOR_TARGET ${TARGET_NAME}.${SIMULATOR_EXTENSION})
set(FPGA_TARGET ${TARGET_NAME}.${FPGA_EXTENSION})
set(IP_EXPORT_TARGET ${TARGET_NAME}.${IP_EXPORT_EXTENSION})

# FPGA device selection
if(DEFINED FPGA_DEVICE)
    message(STATUS "Configuring the design to run on ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    # add qactypes to link command on Windows only
    set(QACTYPES_COMPILE "/Qactypes")
    set(QACTYPES_LINK "/Qactypes")
    set(WIN_FLAG "/EHsc")
else()
    # add qactypes for Linux
    set(QACTYPES_COMPILE "-qactypes")
    set(QACTYPES_LINK "-qactypes")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate
#    representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking. For
#    this reason, FPGA backend flags must be passed as link flags in CMake.

set(EMULATOR_COMPILE_FLAGS    "-fsycl -fintelfpga -DFPGA_EMULATOR  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(EMULATOR_LINK_FLAGS       "-fsycl -fintelfpga -DFPGA_EMULATOR  ${QACTYPES_LINK}    ${USER_FLAGS}                                    ")
set(REPORT_COMPILE_FLAGS      "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(REPORT_LINK_FLAGS         "-fsycl -fintelfpga -DFPGA_HARDWARE                      ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xstarget=${FPGA_DEVICE} -Xshardware -fsycl-link=early")
set(SIMULATOR_COMPILE_FLAGS   "-fsycl -fintelfpga -DFPGA_SIMULATOR ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(SIMULATOR_LINK_FLAGS      "-fsycl -fintelfpga -DFPGA_SIMULATOR ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xstarget=${FPGA_DEVICE} -Xssimulation -Xsghdl=1")
set(FULL_SYSTEM_COMPILE_FLAGS "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(FULL_SYSTEM_LINK_FLAGS    "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xstarget=${FPGA_DEVICE} -Xshardware")

# get rid of this once host pipes work properly
set(IP_EXPORT_COMPILE_FLAGS   "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(IP_EXPORT_LINK_FLAGS      "-fsycl -fintelfpga -DFPGA_HARDWARE                      ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xstarget=${FPGA_DEVICE} -Xshardware -fsycl-link=early -fsycl-device-code-split=per_kernel")

# remove extra spaces
string(REGEX REPLACE "( +)" " " EMULATOR_COMPILE_FLAGS     ${EMULATOR_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " EMULATOR_LINK_FLAGS        ${EMULATOR_LINK_FLAGS})
string(REGEX REPLACE "( +)" " " REPORT_COMPILE_FLAGS       ${REPORT_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " REPORT_LINK_FLAGS          ${REPORT_LINK_FLAGS})
string(REGEX REPLACE "( +)" " " SIMULATOR_COMPILE_FLAGS    ${SIMULATOR_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " SIMULATOR_LINK_FLAGS       ${SIMULATOR_LINK_FLAGS})
string(REGEX REPLACE "( +)" " " FULL_SYSTEM_COMPILE_FLAGS  ${FULL_SYSTEM_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " FULL_SYSTEM_COMPILE_FLAGS  ${FULL_SYSTEM_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " IP_EXPORT_COMPILE_FLAGS    ${IP_EXPORT_COMPILE_FLAGS})
string(REGEX REPLACE "( +)" " " IP_EXPORT_LINK_FLAGS       ${IP_EXPORT_LINK_FLAGS})
###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    icpx -fsycl -fintelfpga fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
add_custom_target(${EMULATOR_EXTENSION} DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

###############################################################################
### Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_compile.cpp -o fpga_compile.report.a
set(REPORT_TARGET ${TARGET_NAME}.report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${REPORT_TARGET} ${SOURCE_FILE})
target_include_directories(${REPORT_TARGET} PRIVATE ../../../include)
add_custom_target(${REPORT_EXTENSION} DEPENDS ${REPORT_TARGET})
set_target_properties(${REPORT_TARGET} PROPERTIES COMPILE_FLAGS "${REPORT_COMPILE_FLAGS}")
set_target_properties(${REPORT_TARGET} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS}")

###############################################################################
### FPGA Simulation
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xstarget=<FPGA_DEVICE> -Xssimulation -Xsghdl=1 fpga_compile.cpp -o fpga_compile.fpga_sim
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    icpx -fsycl -fintelfpga -Xstarget=<FPGA_DEVICE> -Xssimulation -Xsghdl=1 fpga_compile.cpp.o -o fpga_compile.fpga_sim
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
add_custom_target(${SIMULATOR_EXTENSION} DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

###############################################################################
### FPGA Board
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp -o fpga_compile.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(${FPGA_EXTENSION} DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${FULL_SYSTEM_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${FULL_SYSTEM_LINK_FLAGS}")

###############################################################################
### FPGA IP Export (only necessary until native host pipes)
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early -fsycl-device-code-split=per_kernel fpga_compile.cpp -o fpga_compile.fpga_ip_export
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga_ip_export
set(IP_EXPORT_TARGET ${TARGET_NAME}.fpga_ip_export.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${IP_EXPORT_TARGET} ${SOURCE_FILE})
target_include_directories(${IP_EXPORT_TARGET} PRIVATE ../../../include)
add_custom_target(${IP_EXPORT_EXTENSION} DEPENDS ${IP_EXPORT_TARGET})
set_target_properties(${IP_EXPORT_TARGET} PROPERTIES COMPILE_FLAGS "${IP_EXPORT_COMPILE_FLAGS}")
set_target_properties(${IP_EXPORT_TARGET} PROPERTIES LINK_FLAGS "${IP_EXPORT_LINK_FLAGS}")