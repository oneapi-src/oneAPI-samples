set(TARGET_NAME db)
set(SOURCE_FILE db.cpp dbdata.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# which query are we doing?
if(NOT DEFINED QUERY)
    message(STATUS "\t!!QUERY variable is NOT set, defaulting to QUERY=1!!")
    set(QUERY 1)
else()
    message(STATUS "\tQUERY=${QUERY}")
endif()

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

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes")
else()
    set(AC_TYPES_FLAG "-qactypes")
endif()

# ensure a supported query was requested
if(NOT ${QUERY} EQUAL 1 AND NOT ${QUERY} EQUAL 11 AND NOT ${QUERY} EQUAL 12)
  message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 11 and 12)")
endif()

# Pick the default seed if the user did not specify one to CMake.
# We do a seed sweep to find a good seed by default
if(NOT DEFINED SEED)
    if(DEVICE_FLAG MATCHES "A10")
        if(${QUERY} EQUAL 1)
            set(SEED "-Xsseed=2")
        elseif(${QUERY} EQUAL 11)
            set(SEED "-Xsseed=4")
        elseif(${QUERY} EQUAL 12)
            set(SEED "-Xsseed=2")
        endif()
    elseif(DEVICE_FLAG MATCHES "S10")
        if(${QUERY} EQUAL 1)
            set(SEED "-Xsseed=3")
        elseif(${QUERY} EQUAL 11)
            set(SEED "-Xsseed=3")
        elseif(${QUERY} EQUAL 12)
            set(SEED "-Xsseed=2")
        endif()
    elseif(DEVICE_FLAG MATCHES "Agilex7")
        if(${QUERY} EQUAL 1)
            set(SEED "-Xsseed=2")
        elseif(${QUERY} EQUAL 11)
            set(SEED "-Xsseed=4")
        elseif(${QUERY} EQUAL 12)
            set(SEED "-Xsseed=3")
        endif()
    else()
        set(SEED "-Xsseed=1")
    endif()
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED "")
endif()

# Error out if trying to run Q11 on Arria 10
if (DEVICE_FLAG MATCHES "A10")
    if(${QUERY} EQUAL 11)
      message(FATAL_ERROR "Query 11 is not supported on Arria 10 devices")
    endif()
endif()

# set number of CPU cores to use for quartus® compile, default=2
if(NOT DEFINED QUARTUS_CORES)
    set(QUARTUS_CORES 2)
endif()

# check if they want to use the small database
if(SF_SMALL)
    message(STATUS "\tManual override for database size - building for small database (SF=0.01)")
    set(SF_SMALL_ARG -DSF_SMALL)
else()
    set(SF_SMALL_ARG )
endif()

# setting source file based on query version
if(${QUERY} EQUAL 1)
    set(DEVICE_SOURCE query1/query1_kernel.cpp)
    set(DEVICE_HEADER query1/query1_kernel.hpp)
elseif(${QUERY} EQUAL 11)
    set(DEVICE_SOURCE query11/query11_kernel.cpp)
    set(DEVICE_HEADER query11/query11_kernel.hpp)
elseif(${QUERY} EQUAL 12)
    set(DEVICE_SOURCE query12/query12_kernel.cpp)
    set(DEVICE_HEADER query12/query12_kernel.hpp)
else()
    message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 11 and 12)")
endif()

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${AC_TYPES_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga -DQUERY=${QUERY} ${SF_SMALL_ARG} ${AC_TYPES_FLAG}")
set(SIMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${AC_TYPES_FLAG} -Xssimulation -DFPGA_SIMULATOR")
set(SIMULATOR_LINK_FLAGS "-fsycl -fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS} ${AC_TYPES_FLAG}")
set(REPORT_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${AC_TYPES_FLAG} -DFPGA_HARDWARE")
set(REPORT_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsparallel=2 -Xsseed=2 -Xstarget=${FPGA_DEVICE} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -Wall ${WIN_FLAG} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${AC_TYPES_FLAG} -DFPGA_HARDWARE")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsparallel=2 ${SEED} -Xstarget=${FPGA_DEVICE} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS} ${AC_TYPES_FLAG}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE})
target_include_directories(${SIMULATOR_TARGET} PRIVATE ../../../include)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS}")
add_custom_target(fpga_sim DEPENDS ${SIMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE} ${DEVICE_SOURCE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ../../../include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${REPORT_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${REPORT_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE} ${DEVICE_SOURCE})
target_include_directories(${FPGA_TARGET} PRIVATE ../../../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.
