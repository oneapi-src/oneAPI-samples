set(SOURCE_FILE loop_initiation_interval.cpp)
set(TARGET_NAME loop_ii)
set(TARGET_NAME_ENABLE_II loop_ii_enable_ii)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(SIMULATOR_TARGET_ENABLE_II ${TARGET_NAME_ENABLE_II}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_ENABLE_II ${TARGET_NAME_ENABLE_II}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    set(DEVICE_FLAG "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
else()
    string(TOLOWER ${FPGA_DEVICE} FPGA_DEVICE_NAME)
    if(FPGA_DEVICE_NAME MATCHES ".*a10.*" OR FPGA_DEVICE_NAME MATCHES ".*arria10.*")
      set(DEVICE_FLAG "A10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*s10.*" OR FPGA_DEVICE_NAME MATCHES ".*stratix10.*")
      set(DEVICE_FLAG "S10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*agilex.*")
      set(DEVICE_FLAG "Agilex7")
    endif()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Please make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or \
                         -DDEVICE_FLAG=Agilex7.")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_EMULATOR -D${DEVICE_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DFPGA_SIMULATOR -D${DEVICE_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xstarget=${FPGA_DEVICE} -D${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -D${DEVICE_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} -D${DEVICE_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR -DA10 loop_initiation_interval.cpp -o loop_ii.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -DA10 -o loop_ii.cpp.o -c loop_initiation_interval.cpp
#    [link]    icpx -fsycl -fintelfpga loop_ii.cpp.o -o loop_ii.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -DA10 -fsycl-link=early loop_initiation_interval.cpp -o loop_ii_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
set(FPGA_EARLY_IMAGE_ENABLE_II ${TARGET_NAME_ENABLE_II}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../../include)
add_executable(${FPGA_EARLY_IMAGE_ENABLE_II} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE_ENABLE_II} PRIVATE ../../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} ${FPGA_EARLY_IMAGE_ENABLE_II})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
set_target_properties(${FPGA_EARLY_IMAGE_ENABLE_II} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DENABLE_II")
set_target_properties(${FPGA_EARLY_IMAGE_ENABLE_II} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR -DA10 loop_initiation_interval.cpp -o loop_ii.fpga_sim
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -DA10 -o loop_ii.cpp.o -c loop_initiation_interval.cpp
#    [link]    icpx -fsycl -fintelfpga -Xssimulation -Xstarget=<FPGA_DEVICE> -DA10 loop_ii.cpp.o -o loop_ii.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../../include)
add_executable(${SIMULATOR_TARGET_ENABLE_II} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET_ENABLE_II} PRIVATE ../../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET} ${SIMULATOR_TARGET_ENABLE_II})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
set_target_properties(${SIMULATOR_TARGET_ENABLE_II} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS} -DENABLE_II")
set_target_properties(${SIMULATOR_TARGET_ENABLE_II} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -DA10 loop_initiation_interval.cpp -o loop_ii.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o loop_initiation_interval.cpp.o -DA10 -c loop_initiation_interval.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -DA10 loop_initiation_interval.cpp.o -o loop_ii.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../../include)
add_executable(${FPGA_TARGET_ENABLE_II} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET_ENABLE_II} PRIVATE ../../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET} ${FPGA_TARGET_ENABLE_II})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
set_target_properties(${FPGA_TARGET_ENABLE_II} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DENABLE_II")
set_target_properties(${FPGA_TARGET_ENABLE_II} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET_ENABLE_II}")

