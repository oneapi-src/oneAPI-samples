set(TARGET_NAME anr)
set(SOURCE_FILE main.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

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
    set(WIN_FLAG "/EHsc")
endif()

# Allow the user to enable hardware profiling
# Profiling can be enabled when running cmake by adding the flag -DPROFILE_HW=1
# e.g. cmake .. -DPROFILE_HW=1
if(PROFILE_HW)
    set(PROFILE_FLAG "-Xsprofile")
endif()

# Allow the user to do a flat compile
# Profiling can be enabled when running cmake by adding the flag -DFLAT_COMPILE=1
# e.g. cmake .. -DFLAT_COMPILE=1
if(FLAG_COMPILE)
    set(FLAG_COMPILE_FLAG "-Xsbsp-flow=flat")
endif()

# Choose the random seed for the hardware compile
# e.g. cmake .. -DSEED=7
if(NOT DEFINED SEED)
    # the default seed
    set(SEED 3)
else()
    message(STATUS "Seed explicitly set to ${SEED}")
endif()

# Allow the user to change the filter size
# e.g. cmake .. -DFILTER_SIZE=9
if(FILTER_SIZE)
    set(FILTER_SIZE_FLAG "-DFILTER_SIZE=${FILTER_SIZE}")
    message(STATUS "FILTER_SIZE explicitly set to ${FILTER_SIZE}")
endif()

# Allow the user to change the pixels per cycle
# e.g. cmake .. -DPIXELS_PER_CYCLE=9
if(PIXELS_PER_CYCLE)
    set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=${PIXELS_PER_CYCLE}")
    message(STATUS "PIXELS_PER_CYCLE explicitly set to ${PIXELS_PER_CYCLE}")
else()
  # Default PIXELS_PER_CYCLE based on the board being used
  if(FPGA_BOARD MATCHES ".*a10.*")
    set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=2")
  elseif(FPGA_BOARD MATCHES ".*s10.*")
    set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=1")
  elseif(FPGA_BOARD MATCHES ".*agilex.*")
    set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=1")
  else()
    set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=1")
    message(WARNING "Unknown board: setting PIXELS_PER_CYCLE to 1")
  endif()
endif()

# Allow the user to change the Fmax target
if(CLOCK_TARGET)
  message(STATUS "CLOCK_TARGET explicitly set to ${CLOCK_TARGET}")
else()
  # Default CLOCK_TARGET based on the board being used
  if(FPGA_BOARD MATCHES ".*a10.*")
    set(CLOCK_TARGET 300MHz)
  elseif(FPGA_BOARD MATCHES ".*s10.*")
    set(CLOCK_TARGET 480MHz)
  elseif(FPGA_BOARD MATCHES ".*agilex.*")
    set(CLOCK_TARGET 480MHz)
  else()
  set(CLOCK_TARGET 300MHz)
    message(WARNING "Unknown board: setting CLOCK_TARGET to 300MHz")
  endif()
endif()

# Allow the user to change the maximum number of pixels per column
# e.g. cmake .. -DMAX_COLS=3840
if(MAX_COLS)
    set(MAX_COLS_FLAG "-DMAX_COLS=${MAX_COLS}")
    message(STATUS "MAX_COLS explicitly set to ${MAX_COLS}")
endif()

# Allow the user to disable connections to global memory
# This mode is used to test the Fmax and area of the standalone IP.
# In this mode, the Validation will not be correct, but 
# e.g. cmake .. -DDISABLE_GLOBAL_MEM
if(DISABLE_GLOBAL_MEM)
    set(DISABLE_GLOBAL_MEM_FLAG "-DDISABLE_GLOBAL_MEM")
    message(STATUS "Global memory access has been manually disabled")
endif()

# Increase the allowable constexpr steps for the front end. This allows the
# front-end compiler to do more compile-time computation.
set(CONSTEXPR_STEPS "-fconstexpr-steps=1271242")

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${CONSTEXPR_STEPS} ${WIN_FLAG} -fintelfpga ${ENABLE_USM} ${FILTER_SIZE_FLAG} ${PIXELS_PER_CYCLE_FLAG} ${MAX_COLS_FLAG} ${DISABLE_GLOBAL_MEM_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${ENABLE_USM} ${FILTER_SIZE_FLAG} ${PIXELS_PER_CYCLE_FLAG} ${MAX_COLS_FLAG} ${DISABLE_GLOBAL_MEM_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${CONSTEXPR_STEPS} ${WIN_FLAG} -fintelfpga ${ENABLE_USM} ${FILTER_SIZE_FLAG} ${PIXELS_PER_CYCLE_FLAG} ${MAX_COLS_FLAG} ${DISABLE_GLOBAL_MEM_FLAG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware ${PROFILE_FLAG} ${FLAG_COMPILE_FLAG} -Xsparallel=2 -Xsseed=${SEED} -Xsclock=${CLOCK_TARGET} -Xsboard=${FPGA_BOARD} ${ENABLE_USM} ${FILTER_SIZE_FLAG} ${PIXELS_PER_CYCLE_FLAG} ${MAX_COLS_FLAG} ${DISABLE_GLOBAL_MEM_FLAG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.