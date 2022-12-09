###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILE fpga_template.cpp)
set(TARGET_NAME fpga_template)

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
    set(FPGA_DEVICE "intel_s10sx_pac:pac_s10_usm")
endif()

# Use cmake -DUSER_FPGA_FLAGS=<flags> to set extra flags for FPGA backend
# compilation. 
set(USER_FPGA_FLAGS "${USER_FPGA_FLAGS}")

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS "${USER_FLAGS}")
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
    set(QACTYPES_LINK "")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate
#    representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking. For
#    this reason, FPGA backend flags must be passed as link flags in CMake.

set(EMULATOR_COMPILE_FLAGS    "-fsycl -fintelfpga -DFPGA_EMULATOR  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(EMULATOR_LINK_FLAGS       "-fsycl -fintelfpga -DFPGA_EMULATOR  ${QACTYPES_LINK}    ${USER_FLAGS}                                    ")
set(REPORT_COMPILE_FLAGS      "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(REPORT_LINK_FLAGS         "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xshardware -fsycl-link=early")
set(SIMULATOR_COMPILE_FLAGS   "-fsycl -fintelfpga -DFPGA_SIMULATOR ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(SIMULATOR_LINK_FLAGS      "-fsycl -fintelfpga -DFPGA_SIMULATOR ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xssimulation -Xsghdl=1")
set(FULL_SYSTEM_COMPILE_FLAGS "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(FULL_SYSTEM_LINK_FLAGS    "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xshardware")

# get rid of this once host pipes work properly
set(IP_EXPORT_COMPILE_FLAGS   "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_COMPILE} ${USER_FLAGS} ${EXPERIMENTAL_INCLUDE} ${WIN_FLAG}")
set(IP_EXPORT_LINK_FLAGS      "-fsycl -fintelfpga -DFPGA_HARDWARE  ${QACTYPES_LINK}    ${USER_FLAGS}                                     ${USER_FPGA_FLAGS} -Xshardware -fsycl-link=early -fsycl-device-code-split=per_kernel")

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
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})

add_custom_target( displayEmulationCompileCommand ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
    COMMENT "${CMAKE_CXX_COMPILER} ${EMULATOR_COMPILE_FLAGS} -o ${OBJ_FILE} -c ${SOURCE_FILE}")

add_dependencies(${EMULATOR_TARGET} displayEmulationCompileCommand)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
add_custom_command(TARGET ${EMULATOR_TARGET} POST_BUILD COMMENT "${CMAKE_CXX_COMPILER} ${EMULATOR_LINK_FLAGS} ${OBJ_FILE} -o ${EMULATOR_TARGET}")

set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(${EMULATOR_EXTENSION} DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Report
###############################################################################
add_executable(${REPORT_TARGET} ${SOURCE_FILE})

add_custom_target( displayReportCompileCommand ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
    COMMENT "${CMAKE_CXX_COMPILER} ${REPORT_COMPILE_FLAGS} -o ${OBJ_FILE} -c ${SOURCE_FILE}")

add_dependencies(${REPORT_TARGET} displayReportCompileCommand)
set_target_properties(${REPORT_TARGET} PROPERTIES COMPILE_FLAGS "${REPORT_COMPILE_FLAGS}")
add_custom_command(TARGET ${REPORT_TARGET} POST_BUILD COMMENT "${CMAKE_CXX_COMPILER} ${REPORT_LINK_FLAGS} ${OBJ_FILE} -o ${REPORT_TARGET}")

set_target_properties(${REPORT_TARGET} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS}")
add_custom_target(${REPORT_EXTENSION} DEPENDS ${REPORT_TARGET})

###############################################################################
### FPGA Simulation
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})

add_custom_target( displaySimulatorCompileCommand ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
    COMMENT "${CMAKE_CXX_COMPILER} ${SIMULATOR_COMPILE_FLAGS} -o ${OBJ_FILE} -c ${SOURCE_FILE}")

add_dependencies(${SIMULATOR_TARGET} displaySimulatorCompileCommand)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
add_custom_command(TARGET ${SIMULATOR_TARGET} POST_BUILD COMMENT "${CMAKE_CXX_COMPILER} ${SIMULATOR_LINK_FLAGS} ${OBJ_FILE} -o ${SIMULATOR_TARGET}")

set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(${SIMULATOR_EXTENSION} DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### FPGA Board
###############################################################################
add_executable(${FPGA_TARGET} ${SOURCE_FILE})

add_custom_target( displayFpgaCompileCommand ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
    COMMENT "${CMAKE_CXX_COMPILER} ${FULL_SYSTEM_COMPILE_FLAGS} -o ${OBJ_FILE} -c ${SOURCE_FILE}")

add_dependencies(${FPGA_TARGET} displayFpgaCompileCommand)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${FULL_SYSTEM_COMPILE_FLAGS}")
add_custom_command(TARGET ${FPGA_TARGET} POST_BUILD COMMENT "${CMAKE_CXX_COMPILER} ${FULL_SYSTEM_LINK_FLAGS} ${OBJ_FILE} -o ${FPGA_TARGET}")

set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${FULL_SYSTEM_LINK_FLAGS}")
add_custom_target(${FPGA_EXTENSION} DEPENDS ${FPGA_TARGET})

###############################################################################
### FPGA IP Export (only necessary until native host pipes)
###############################################################################
add_executable(${IP_EXPORT_TARGET} ${SOURCE_FILE})

add_custom_target( displayExportCompileCommand ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
    COMMENT "${CMAKE_CXX_COMPILER} ${IP_EXPORT_COMPILE_FLAGS} -o ${OBJ_FILE} -c ${SOURCE_FILE}")

add_dependencies(${IP_EXPORT_TARGET} displayExportCompileCommand)
set_target_properties(${IP_EXPORT_TARGET} PROPERTIES COMPILE_FLAGS "${IP_EXPORT_COMPILE_FLAGS}")
add_custom_command(TARGET ${IP_EXPORT_TARGET} POST_BUILD COMMENT "${CMAKE_CXX_COMPILER} ${IP_EXPORT_LINK_FLAGS} ${OBJ_FILE} -o ${IP_EXPORT_TARGET}")

set_target_properties(${IP_EXPORT_TARGET} PROPERTIES LINK_FLAGS "${IP_EXPORT_LINK_FLAGS}")
add_custom_target(${IP_EXPORT_EXTENSION} DEPENDS ${IP_EXPORT_TARGET})