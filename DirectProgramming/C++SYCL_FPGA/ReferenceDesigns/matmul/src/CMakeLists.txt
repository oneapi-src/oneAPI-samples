set(TARGET_NAME matmul)
set(HOST_SOURCE_FILE matmul_demo.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_EARLY_IMAGE ${TARGET_NAME}.report.a)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    set(DEVICE_FLAG "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")

    set(BSP_FLAG "")
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

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1" OR FPGA_DEVICE MATCHES ".*pac_a10.*|.*pac_s10.*")
        set(BSP_FLAG "-DIS_BSP")
    else()
        set(BSP_FLAG "")
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number, so the IS_BSP macro will not be passed to your C++ code.")
        message(STATUS "If the target is actually a BSP, run cmake with -DIS_BSP=1 to pass the IS_BSP macro to your C++ code.")
    endif()
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "/EHsc /Qactypes /Wall ")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "/Qactypes ")
    set(EMULATOR_PLATFORM_FLAGS "/fp:precise")
else()
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "-qactypes -Wall -fno-finite-math-only ")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "")
    set(EMULATOR_PLATFORM_FLAGS "")
endif()

# Set design parameters according to the selected board
if(DEVICE_FLAG MATCHES "A10")
    # A10 parameters
    set(ROWS_A 64)
    set(COMMON 64)
    set(COLS_B 64)
    set(TILE_A 8)
    set(TILE_B 8)
    set(CLOCK_TARGET 360MHz)
    set(SEED "-Xsseed=7")
elseif(DEVICE_FLAG MATCHES "S10")
    # S10 parameters
    set(ROWS_A 64)
    set(COMMON 64)
    set(COLS_B 64)
    set(TILE_A 8)
    set(TILE_B 8)
    set(CLOCK_TARGET 480MHz)
    set(SEED "-Xsseed=2")
elseif(DEVICE_FLAG MATCHES "Agilex")
    # Agilex™ parameters
    set(ROWS_A 64)
    set(COMMON 64)
    set(COLS_B 64)
    set(TILE_A 8)
    set(TILE_B 8)
    set(CLOCK_TARGET 540MHz)
    set(SEED "-Xsseed=5")
else()
    message(FATAL_ERROR "An incorrect DEVICE_FLAG was given. Make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or -DDEVICE_FLAG=Agilex7.")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

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

message(STATUS "ROWS_A=${ROWS_A}")
message(STATUS "COMMON=${COMMON}")
message(STATUS "COLS_B=${COLS_B}")
message(STATUS "TILE_A=${TILE_A}")
message(STATUS "TILE_B=${TILE_B}")
message(STATUS "SEED=${SEED}")

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga ${EMULATOR_PLATFORM_FLAGS} ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -DROWS_A=${ROWS_A} -DCOMMON=${COMMON} -DCOLS_B=${COLS_B} -DTILE_A=${TILE_A} -DTILE_B=${TILE_B} -DFPGA_EMULATOR ${BSP_FLAG} ${USER_FLAGS}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${EMULATOR_PLATFORM_FLAGS} ${PLATFORM_SPECIFIC_LINK_FLAGS} ${BSP_FLAG} ${USER_FLAGS}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -DFPGA_SIMULATOR -DROWS_A=${ROWS_A} -DCOMMON=${COMMON} -DCOLS_B=${COLS_B} -DTILE_A=${TILE_A} -DTILE_B=${TILE_B} ${USER_HARDWARE_FLAGS} ${BSP_FLAG} ${USER_FLAGS}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xssimulation -Xsghdl -Xsclock=${CLOCK_TARGET} -Xstarget=${FPGA_DEVICE} ${USER_SIMULATOR_FLAGS} ${BSP_FLAG} ${USER_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -DROWS_A=${ROWS_A} -DCOMMON=${COMMON} -DCOLS_B=${COLS_B} -DTILE_A=${TILE_A} -DTILE_B=${TILE_B} -DFPGA_HARDWARE ${BSP_FLAG} ${USER_FLAGS}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xshardware -Xsclock=${CLOCK_TARGET} ${SEED} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${BSP_FLAG} ${USER_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${HOST_SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${HOST_SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${HOST_SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${HOST_SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.