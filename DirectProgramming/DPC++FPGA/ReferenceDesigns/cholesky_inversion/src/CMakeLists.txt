set(TARGET_NAME cholesky_inversion)
set(SOURCE_FILE cholesky_inversion_demo.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "/EHsc /Qactypes /Wall")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "/Qactypes")
else()
    set(PLATFORM_SPECIFIC_COMPILE_FLAGS "-qactypes -Wall")
    set(PLATFORM_SPECIFIC_LINK_FLAGS "")
endif()


# A10 parameters
set(MATRIX_DIMENSION 32)
set(COMPLEX 0)
set(FIXED_ITERATIONS_DECOMPOSITION 39)
set(FIXED_ITERATIONS_INVERSION 34)
set(CLOCK_TARGET 360MHz)
set(SEED "-Xsseed=29")

# Set design parameters according to the selected board
if(FPGA_BOARD MATCHES ".*a10.*")
    # A10 parameters
    # Nothing to do
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(MATRIX_DIMENSION 32)
    set(COMPLEX 0)
    set(FIXED_ITERATIONS_DECOMPOSITION 44)
    set(FIXED_ITERATIONS_INVERSION 44)
    set(CLOCK_TARGET 450MHz)
    set(SEED "-Xsseed=5")
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™ parameters
    set(MATRIX_DIMENSION 32)
    set(FIXED_ITERATIONS_DECOMPOSITION 45)
    set(FIXED_ITERATIONS_INVERSION 45)
    set(COMPLEX 0)
    set(CLOCK_TARGET 600MHz)
    set(SEED "-Xsseed=5")
else()
    message(STATUS "Unknown board ${FPGA_BOARD}!")
    message(STATUS "Using Arria 10 defaults.")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

if(DEFINED ENV{MATRIX_DIMENSION})
    set(MATRIX_DIMENSION $ENV{MATRIX_DIMENSION})
endif()

if(DEFINED ENV{COMPLEX})
    set(COMPLEX $ENV{COMPLEX})
endif()

if(DEFINED ENV{FIXED_ITERATIONS_DECOMPOSITION})
    set(FIXED_ITERATIONS_DECOMPOSITION $ENV{FIXED_ITERATIONS_DECOMPOSITION})
endif()

if(DEFINED ENV{FIXED_ITERATIONS_INVERSION})
    set(FIXED_ITERATIONS_INVERSION $ENV{FIXED_ITERATIONS_INVERSION})
endif()

message(STATUS "MATRIX_DIMENSION=${MATRIX_DIMENSION}")
message(STATUS "COMPLEX=${COMPLEX}")
message(STATUS "FIXED_ITERATIONS_DECOMPOSITION=${FIXED_ITERATIONS_DECOMPOSITION}")
message(STATUS "FIXED_ITERATIONS_INVERSION=${FIXED_ITERATIONS_INVERSION}")
message(STATUS "SEED=${SEED}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -fbracket-depth=512 -fintelfpga -fno-finite-math-only -DFIXED_ITERATIONS_DECOMPOSITION=${FIXED_ITERATIONS_DECOMPOSITION} -DFIXED_ITERATIONS_INVERSION=${FIXED_ITERATIONS_INVERSION} -DCOMPLEX=${COMPLEX} -DMATRIX_DIMENSION=${MATRIX_DIMENSION} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${PLATFORM_SPECIFIC_COMPILE_FLAGS} -Wformat-security -Werror=format-security -fintelfpga -fbracket-depth=512 -DFIXED_ITERATIONS_DECOMPOSITION=${FIXED_ITERATIONS_DECOMPOSITION} -DFIXED_ITERATIONS_INVERSION=${FIXED_ITERATIONS_INVERSION} -DCOMPLEX=${COMPLEX} -DMATRIX_DIMENSION=${MATRIX_DIMENSION} -fp-model=precise -Xsfp-relaxed")
set(HARDWARE_LINK_FLAGS "-fintelfpga ${PLATFORM_SPECIFIC_LINK_FLAGS} -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 ${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS} -fp-model=precise -Xsfp-relaxed")
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
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
