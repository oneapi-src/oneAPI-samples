###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILE add.cpp)
set(TARGET_NAME add)

# When generating a standalone IP, the `FPGA_DEVICE` will be targeted. You can
# specify a device family (E.g. "Arria10" or "Stratix10") or a specific part
# number (E.g. "10AS066N3F40E2SG")
set(FPGA_DEVICE "10AS066N3F40E2SG")

# When targeting a board, the `FPGA_BOARD` will be targeted. Options are:
#   intel_s10sx_pac:pac_s10
#   intel_s10sx_pac:pac_s10_usm
#   intel_a10gx_pac:pac_a10
set(FPGA_BOARD "intel_s10sx_pac:pac_s10_usm")

set(USER_HARDWARE_FLAGS "")
# use cmake -DUSER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend
# compilation

set(IPA_EXPERIMENTAL_INCLUDE "-I\"$ENV{INTELFPGAOCLSDKROOT}/include\"")
set(USER_FLAGS "${IPA_EXPERIMENTAL_INCLUDE}")
# use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation

###############################################################################
### no changes after here
###############################################################################

# define the different build targets the generated build script will support.
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)

# IP Authoring
set(IP_REPORT_TARGET ${TARGET_NAME}.fpga_ip_report)
set(IP_SIMULATOR_TARGET ${TARGET_NAME}.fpga_ip_sim)
set(IP_TARGET ${TARGET_NAME}.fpga_ip)
set(IP_QII_TARGET ${TARGET_NAME}.fpga_ip_qii)

# Full-system
set(BOARD_REPORT_TARGET ${TARGET_NAME}.fpga_board_report)
set(BOARD_SIMULATOR_TARGET ${TARGET_NAME}.fpga_board_sim)
set(BOARD_TARGET ${TARGET_NAME}.fpga_board)

# FPGA board selection
if(DEFINED FPGA_BOARD)
    message(STATUS "Configuring the design to run on FPGA accelerator board ${FPGA_BOARD}")
endif()

if(DEFINED FPGA_DEVICE)
    message(STATUS "Configuring the design to compile IP for FPGA device ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
    # add qactypes to link command on Windows only
    set(QACTYPES_LINK "-qactypes")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate
#    representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking. For
#    this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -Wall ${WIN_FLAG} -qactypes")
set(EMULATOR_LINK_FLAGS    "-fintelfpga ${QACTYPES_LINK}")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga -Wall ${WIN_FLAG} -qactypes")
set(BOARD_LINK_FLAGS       "-fintelfpga -Xsboard=${FPGA_BOARD}   ${USER_HARDWARE_FLAGS} ${QACTYPES_LINK}")
set(IP_LINK_FLAGS          "-fintelfpga -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${QACTYPES_LINK}")

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    dpcpp -fintelfpga fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS} -DFPGA_EMULATOR ${USER_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS} -DFPGA_EMULATOR ${USER_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### IP Authoring Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_compile.cpp -o fpga_compile_report.a
set(IP_REPORT_TARGET ${TARGET_NAME}.fpga_ip_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${IP_REPORT_TARGET} ${SOURCE_FILE})
set_target_properties(${IP_REPORT_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_HARDWARE ${USER_FLAGS}")
set_target_properties(${IP_REPORT_TARGET} PROPERTIES LINK_FLAGS "${IP_LINK_FLAGS} ${USER_FLAGS} -Xshardware -fsycl-link=early")
add_custom_target(fpga_ip_report DEPENDS ${IP_REPORT_TARGET})
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### IP Authoring Simulation
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xstarget=<FPGA_DEVICE> -Xssimulation -ghdl=1 fpga_compile.cpp -o fpga_compile.fpga_sim
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    dpcpp -fintelfpga -Xstarget=<FPGA_DEVICE> -Xssimulation -Xsghdl=1 fpga_compile.cpp.o -o fpga_compile.fpga_sim
add_executable(${IP_SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_ip_sim DEPENDS ${IP_SIMULATOR_TARGET})
set_target_properties(${IP_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_SIMULATOR ${USER_FLAGS}")
set_target_properties(${IP_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${IP_LINK_FLAGS} -Xssimulation -Xsghdl=1 ${USER_FLAGS}") # don't use code-split here because it doesn't work for sim

###############################################################################
### IP Authoring IP Export
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early -fsycl-device-code-split=per_kernel fpga_compile.cpp -o fpga_compile.fpga_ip
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga_ip
add_executable(${IP_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_ip DEPENDS ${IP_TARGET})
set_target_properties(${IP_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_HARDWARE ${USER_FLAGS}")
set_target_properties(${IP_TARGET} PROPERTIES LINK_FLAGS "${IP_LINK_FLAGS} -Xshardware -fsycl-link=early -fsycl-device-code-split=per_kernel ${USER_FLAGS}")

###############################################################################
### IP Authoring Place & Route
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp -o fpga_compile.fpga_ip_qii
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -fsycl-device-code-split=per_kernel -Xstarget=<FPGA_DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga_ip_qii
add_executable(${IP_QII_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_ip_qii DEPENDS ${IP_QII_TARGET})
set_target_properties(${IP_QII_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_HARDWARE ${USER_FLAGS}")
set_target_properties(${IP_QII_TARGET} PROPERTIES LINK_FLAGS "${IP_LINK_FLAGS} -fsycl-device-code-split=per_kernel -Xshardware ${USER_FLAGS}")

###############################################################################
### Board Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_BOARD> -fsycl-link=early fpga_compile.cpp -o fpga_compile_report.a
set(BOARD_REPORT_TARGET ${TARGET_NAME}.fpga_board_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${BOARD_REPORT_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_board_report DEPENDS ${BOARD_REPORT_TARGET})
set_target_properties(${BOARD_REPORT_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_HARDWARE ${USER_FLAGS}")
set_target_properties(${BOARD_REPORT_TARGET} PROPERTIES LINK_FLAGS "${BOARD_LINK_FLAGS} ${USER_FLAGS} -Xshardware -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Board Simulation
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xstarget=<FPGA_BOARD> -Xssimulation -Xsghdl=1 fpga_compile.cpp -o fpga_compile.fpga_sim
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    dpcpp -fintelfpga -Xstarget=<FPGA_BOARD> -Xssimulation -Xsghdl=1 fpga_compile.cpp.o -o fpga_compile.fpga_sim
add_executable(${BOARD_SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_board_sim DEPENDS ${BOARD_SIMULATOR_TARGET})
set_target_properties(${BOARD_SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_SIMULATOR ${USER_FLAGS}")
set_target_properties(${BOARD_SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${BOARD_LINK_FLAGS} -Xssimulation -Xsghdl=1 ${USER_FLAGS}")

###############################################################################
### FPGA Board
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_BOARD> fpga_compile.cpp -o fpga_compile.fpga_board
# CMake executes:
#   [compile] dpcpp -fintelfpga -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_BOARD> fpga_compile.cpp.o -o fpga_compile.fpga_board
add_executable(${BOARD_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga_board DEPENDS ${BOARD_TARGET})
set_target_properties(${BOARD_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFPGA_HARDWARE ${USER_FLAGS}")
set_target_properties(${BOARD_TARGET} PROPERTIES LINK_FLAGS "${BOARD_LINK_FLAGS} -Xshardware ${USER_FLAGS}")
