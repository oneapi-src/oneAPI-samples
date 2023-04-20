set(TARGET_NAME decompress)
set(SOURCE_FILE main.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

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

# Select between SNAPPY and GZIP decompression
if(DEFINED SNAPPY AND DEFINED GZIP)
  message(FATAL_ERROR "Cannot compile for both SNAPPY and GZIP compression. You must define exactly one of -DSNAPPY=1 and -DGZIP=1")
else()
  # check whether user wants GZIP or SNAPPY decompression - default is SNAPPY
  if(DEFINED GZIP)
    message(STATUS "Compiling for GZIP decompression")
    set(DECOMPRESS_FORMAT_FLAG "-DGZIP")
  else()
    if(NOT DEFINED SNAPPY)
      message(STATUS "Neither GZIP nor SNAPPY was defined. Defaulting to SNAPPY decompression")
    endif()
    message(STATUS "Compiling for SNAPPY decompression")
    set(DECOMPRESS_FORMAT_FLAG "-DSNAPPY")
  endif()

  set(ignoreMe "${GZIP}${SNAPPY}")
endif()

# These are Windows-specific flags:
# 1. /EHsc This is a Windows-specific flag that enables exception handling in host code
# 2. /Qactypes Include ac_types headers and link against ac_types emulation libraries
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes")
    # Increase the allowable constexpr steps for the front end. This allows the
    # front-end compiler to do more compile-time computation.
    set(CONSTEXPR_STEPS "/constexpr:steps5084968")
else()
    set(AC_TYPES_FLAG "-qactypes")
    set(CONSTEXPR_STEPS "-fconstexpr-steps=5084968")

endif()

# Allow the user to enable hardware profiling
# Profiling can be enabled when running cmake by adding the flag -DPROFILE_HW=1
# e.g. cmake .. -DPROFILE_HW=1
if(PROFILE_HW)
    set(PROFILE_FLAG "-Xsprofile")
endif()

# Allow the user to do a flat compile
# Compiling a flat version of the design (as opposed to with partial reconfiguration)
# can be enabled when running cmake by adding the flag -DFLAT_COMPILE=1
# e.g. cmake .. -DFLAT_COMPILE=1
if(FLAT_COMPILE)
    set(FLAT_COMPILE_FLAG "-Xsbsp-flow=flat")
endif()

if(IGNORE_DEFAULT_SEED)
  set(SEED_FLAG "")
else()
  if (NOT DEFINED SEED)
    # the default seed for each FPGA type
    if(DEVICE_FLAG MATCHES "A10")
      set(SEED 1)
    elseif(DEVICE_FLAG MATCHES "S10")
      set(SEED 2)
    elseif(DEVICE_FLAG MATCHES "Agilex7")
      set(SEED 3)
    else()
      message(STATUS "SEED not defined and no known seed for this board -- defaulting to SEED = 1")
      set(SEED 1)
    endif()
  endif()

  set(SEED_FLAG "-Xsseed=${SEED}")
endif()

# Allow the user to set how many literals are processed at once
# e.g. cmake .. -DLITERALS_PER_CYCLE=2
if(DEFINED LITERALS_PER_CYCLE)
  set(LITERALS_PER_CYCLE_FLAG "-DLITERALS_PER_CYCLE=${LITERALS_PER_CYCLE}")
endif()


# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${CONSTEXPR_STEPS} ${WIN_FLAG} ${AC_TYPES_FLAG} ${LITERALS_PER_CYCLE_FLAG} ${DECOMPRESS_FORMAT_FLAG} -DFPGA_EMULATOR ${BSP_FLAG}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} ${BSP_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${CONSTEXPR_STEPS} ${WIN_FLAG} ${AC_TYPES_FLAG} ${LITERALS_PER_CYCLE_FLAG} ${DECOMPRESS_FORMAT_FLAG} -DFPGA_SIMULATOR ${BSP_FLAG}")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${AC_TYPES_FLAG} ${BSP_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${CONSTEXPR_STEPS} ${WIN_FLAG} ${AC_TYPES_FLAG} ${LITERALS_PER_CYCLE_FLAG} ${DECOMPRESS_FORMAT_FLAG} -DFPGA_HARDWARE ${BSP_FLAG}")
set(REPORT_LINK_FLAGS "-fsycl -fintelfpga -Xshardware ${PROFILE_FLAG} ${FLAT_COMPILE_FLAG} -Xsparallel=2 ${SEED_FLAG} -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} ${BSP_FLAG}")
set(HARDWARE_LINK_FLAGS "${REPORT_LINK_FLAGS} ${AC_TYPES_FLAG} ${BSP_FLAG}")
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
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

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
