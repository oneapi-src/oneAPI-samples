set(TARGET_NAME db)
set(SOURCE_FILE db.cpp dbdata.cpp)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# which query are we doing?
if(NOT DEFINED QUERY)
    message(STATUS "\t!!QUERY variable is NOT set, defaulting to QUERY=1!!")
    set(QUERY 1)
else()
    message(STATUS "\tQUERY=${QUERY}")
endif()

# select default board based on query
if(${QUERY} EQUAL 1)
    set(DEFAULT_BOARD "intel_a10gx_pac:pac_a10")
    set(DEFAULT_BOARD_STR "Intel Arria(R) 10 GX")
elseif(${QUERY} EQUAL 9)
    set(DEFAULT_BOARD "intel_s10sx_pac:pac_s10")
    set(DEFAULT_BOARD_STR "Intel Stratix(R) 10 SX")
elseif(${QUERY} EQUAL 11)
    set(DEFAULT_BOARD "intel_s10sx_pac:pac_s10")
    set(DEFAULT_BOARD_STR "Intel Stratix(R) 10 SX")
elseif(${QUERY} EQUAL 12)
    set(DEFAULT_BOARD "intel_a10gx_pac:pac_a10")
    set(DEFAULT_BOARD_STR "Intel Arria(R) 10 GX")
endif()

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD ${DEFAULT_BOARD})
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with ${DEFAULT_BOARD_STR} FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# This is a Windows-specific flag that enables error handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# ensure a supported query was requested
if(NOT ${QUERY} EQUAL 1 AND NOT ${QUERY} EQUAL 9 AND NOT ${QUERY} EQUAL 11 AND NOT ${QUERY} EQUAL 12)
  message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 9, 11 and 12)")
endif()

# Pick the default seed if the user did not specify one to CMake.
# We do a seed sweep to find a good seed by default
if(NOT DEFINED SEED)
    if(${FPGA_BOARD} MATCHES ".*a10.*")
        if(${QUERY} EQUAL 1)
            set(SEED 2)
        elseif(${QUERY} EQUAL 9)
            set(SEED 2)
        elseif(${QUERY} EQUAL 11)
            set(SEED 4)
        elseif(${QUERY} EQUAL 12)
            set(SEED 2)
        endif()
    elseif(${FPGA_BOARD} MATCHES ".*s10.*")
        if(${QUERY} EQUAL 1)
            set(SEED 3)
        elseif(${QUERY} EQUAL 9)
            set(SEED 2)
        elseif(${QUERY} EQUAL 11)
            set(SEED 3)
        elseif(${QUERY} EQUAL 12)
            set(SEED 2)
        endif()
    elseif(${FPGA_BOARD} MATCHES ".*agilex.*")
        if(${QUERY} EQUAL 1)
            set(SEED 2)
        elseif(${QUERY} EQUAL 9)
            set(SEED 2)
        elseif(${QUERY} EQUAL 11)
            set(SEED 4)
        elseif(${QUERY} EQUAL 12)
            set(SEED 3)
        endif()
    else()
        set(SEED 1)
    endif()
endif()

# Error out if trying to run Q9 or Q11 on Arria 10
if (${FPGA_BOARD} MATCHES ".*a10.*")
    if(${QUERY} EQUAL 9 OR ${QUERY} EQUAL 11)
      message(FATAL_ERROR "Queries 9 and 11 are not supported on Arria 10 devices")
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
elseif(${QUERY} EQUAL 9)
    set(DEVICE_SOURCE query9/query9_kernel.cpp)
    set(DEVICE_HEADER query9/query9_kernel.hpp)
elseif(${QUERY} EQUAL 11)
    set(DEVICE_SOURCE query11/query11_kernel.cpp)
    set(DEVICE_HEADER query11/query11_kernel.hpp)
elseif(${QUERY} EQUAL 12)
    set(DEVICE_SOURCE query12/query12_kernel.cpp)
    set(DEVICE_HEADER query12/query12_kernel.hpp)
else()
    message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 9, 11 and 12)")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DQUERY=${QUERY} ${SF_SMALL_ARG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga -DQUERY=${QUERY} ${SF_SMALL_ARG}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fintelfpga -DQUERY=${QUERY} ${SF_SMALL_ARG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsparallel=2 -Xsseed=2 -Xsboard=${FPGA_BOARD} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE} ${DEVICE_SOURCE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE} ${DEVICE_SOURCE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE} ${DEVICE_SOURCE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
