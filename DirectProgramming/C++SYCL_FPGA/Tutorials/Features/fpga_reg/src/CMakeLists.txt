set(SOURCE_FILE fpga_reg.cpp)
set(TARGET_NAME fpga_reg)
set(TARGET_NAME_REG fpga_reg_registered)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(SIMULATOR_TARGET_REG ${TARGET_NAME_REG}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_REG ${TARGET_NAME_REG}.fpga)

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
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    icpx -fsycl -fintelfpga fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_reg.cpp -o fpga_reg_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
set(FPGA_EARLY_IMAGE_REG ${TARGET_NAME_REG}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_executable(${FPGA_EARLY_IMAGE_REG} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_REG} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} ${FPGA_EARLY_IMAGE_REG})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

set_target_properties(${FPGA_EARLY_IMAGE_REG} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DUSE_FPGA_REG")
set_target_properties(${FPGA_EARLY_IMAGE_REG} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> fpga_reg.cpp -o fpga_reg.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -o fpga_reg.cpp.o -c fpga_reg.cpp
#   [link]    icpx -fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> fpga_reg.cpp.o -o fpga_reg.fpga
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
add_executable(${SIMULATOR_TARGET_REG} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET_REG} PRIVATE ../../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET} ${SIMULATOR_TARGET_REG})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
set_target_properties(${SIMULATOR_TARGET_REG} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS} -DUSE_FPGA_REG")
set_target_properties(${SIMULATOR_TARGET_REG} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET_REG}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_reg.cpp -o fpga_reg.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o fpga_reg.cpp.o -c fpga_reg.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_reg.cpp.o -o fpga_reg.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_executable(${FPGA_TARGET_REG} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_REG} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET} ${FPGA_TARGET_REG})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
set_target_properties(${FPGA_TARGET_REG} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DUSE_FPGA_REG")
set_target_properties(${FPGA_TARGET_REG} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_REG}")
