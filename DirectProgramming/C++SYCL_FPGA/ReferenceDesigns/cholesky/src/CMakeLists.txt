set(TARGET_NAME cholesky)
set(SOURCE_FILE cholesky_demo.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)

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

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or -DDEVICE_FLAG=Agilex7.")
endif()

if(DEVICE_FLAG MATCHES "A10")
    # A10 parameters
    set(MATRIX_DIMENSION 32)
    set(COMPLEX 0)
    set(FIXED_ITERATIONS 39)
    set(CLOCK_TARGET 360MHz)
    set(SEED "-Xsseed=29")
elseif(DEVICE_FLAG MATCHES "S10")
    # S10 parameters
    set(MATRIX_DIMENSION 32)
    set(COMPLEX 0)
    set(FIXED_ITERATIONS 44)
    set(CLOCK_TARGET 450MHz)
    set(SEED "-Xsseed=5")
elseif(DEVICE_FLAG MATCHES "Agilex7")
    # Agilex 7 parameters
    set(MATRIX_DIMENSION 32)
    set(FIXED_ITERATIONS 45)
    set(COMPLEX 0)
    set(CLOCK_TARGET 520MHz)
    set(SEED "-Xsseed=5")
else()
    message(FATAL_ERROR "An incorrect DEVICE_FLAG was given. Make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or -DDEVICE_FLAG=Agilex7.")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "/EHsc /Qactypes /Wall")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "/Qactypes ")
    set(EMULATOR_PLATFORM_FLAGS "/fp:precise")
else()
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "-qactypes -Wall -fno-finite-math-only ")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "")
    set(EMULATOR_PLATFORM_FLAGS "")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

if(DEFINED SET_MATRIX_DIMENSION)
    set(MATRIX_DIMENSION ${SET_MATRIX_DIMENSION})
endif()

if(DEFINED SET_COMPLEX)
    set(COMPLEX ${SET_COMPLEX})
endif()

if(DEFINED SET_FIXED_ITERATIONS)
    set(FIXED_ITERATIONS ${SET_FIXED_ITERATIONS})
endif()

message(STATUS "MATRIX_DIMENSION=${MATRIX_DIMENSION}")
message(STATUS "COMPLEX=${COMPLEX}")
message(STATUS "FIXED_ITERATIONS=${FIXED_ITERATIONS}")
message(STATUS "SEED=${SEED}")

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${EMULATOR_PLATFORM_FLAGS} ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -fbracket-depth=512 -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DMATRIX_DIMENSION=${MATRIX_DIMENSION} -DFPGA_EMULATOR ${BSP_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${EMULATOR_PLATFORM_FLAGS} ${PLATFORM_SPECIFIC_LINK_FLAGS} ${BSP_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -DFPGA_SIMULATOR -fbracket-depth=512 -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DMATRIX_DIMENSION=${MATRIX_DIMENSION} ${USER_SIMULATOR_FLAGS} ${BSP_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xssimulation -Xsghdl -Xsclock=${CLOCK_TARGET} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -fbracket-depth=512 -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DMATRIX_DIMENSION=${MATRIX_DIMENSION} -DFPGA_HARDWARE ${BSP_FLAG}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 ${SEED} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation


###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
