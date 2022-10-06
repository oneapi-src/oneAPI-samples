set(TARGET_NAME qrd)
set(SOURCE_FILE qrd_demo.cpp)
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

# Include ac_types headers and link against ac_types emulation libraries
set(AC_TYPES_FLAG "-qactypes")

# Setting the stack size on Windows as the default one is too small for this program
if(WIN32)
    set(WIN_LINK_FLAG "-Wl,-stack:12582912")
endif()

# A10 parameters
set(ROWS_COMPONENT 128)
set(COLS_COMPONENT 128)
set(COMPLEX 1)
set(FIXED_ITERATIONS 64)
set(CLOCK_TARGET 360MHz)
set(SEED "-Xsseed=7")
# Overwrite design parameters according to the selected board
if(FPGA_BOARD MATCHES ".*a10.*")
    # A10 parameters
    # Nothing to do
elseif(FPGA_BOARD MATCHES ".*s10.*")
    # S10 parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(COMPLEX 1)
    set(FIXED_ITERATIONS 110)
    set(CLOCK_TARGET 480MHz)
    set(SEED "-Xsseed=22")
elseif(FPGA_BOARD MATCHES ".*agilex.*")
    # Agilex™ parameters
    set(ROWS_COMPONENT 256)
    set(COLS_COMPONENT 256)
    set(FIXED_ITERATIONS 110)
    set(COMPLEX 1)
    set(CLOCK_TARGET 600MHz)
    set(SEED "-Xsseed=5")
else()
    message(STATUS "Unknown board ${FPGA_BOARD}!")
    message(STATUS "Using Arria 10 defaults.")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

if(DEFINED SET_ROWS_COMPONENT)
    set(ROWS_COMPONENT ${SET_ROWS_COMPONENT})
endif()

if(DEFINED SET_COLS_COMPONENT)
    set(COLS_COMPONENT ${SET_COLS_COMPONENT})
endif()

if(DEFINED SET_COMPLEX)
    set(COMPLEX ${SET_COMPLEX})
endif()

if(DEFINED SET_FIXED_ITERATIONS)
    set(FIXED_ITERATIONS ${SET_FIXED_ITERATIONS})
endif()

message(STATUS "ROWS_COMPONENT=${ROWS_COMPONENT}")
message(STATUS "COLS_COMPONENT=${COLS_COMPONENT}")
message(STATUS "COMPLEX=${COMPLEX}")
message(STATUS "FIXED_ITERATIONS=${FIXED_ITERATIONS}")
message(STATUS "SEED=${SEED}")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -Wall ${AC_TYPES_FLAG} -Wformat-security -Werror=format-security -fbracket-depth=512 -fno-finite-math-only -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${AC_TYPES_FLAG} ${WIN_LINK_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga -Wall ${AC_TYPES_FLAG} -Wformat-security -Werror=format-security -fbracket-depth=512 -DFIXED_ITERATIONS=${FIXED_ITERATIONS} -DCOMPLEX=${COMPLEX} -DROWS_COMPONENT=${ROWS_COMPONENT} -DCOLS_COMPONENT=${COLS_COMPONENT}")
set(HARDWARE_LINK_FLAGS "-fintelfpga ${AC_TYPES_FLAG} ${WIN_LINK_FLAG} -Xshardware -Xsclock=${CLOCK_TARGET} -Xsparallel=2 ${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
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
