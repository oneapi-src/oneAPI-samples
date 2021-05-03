set(HOST_SOURCE db.cpp dbdata.cpp)

set(TARGET_NAME db)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# which query are we doing?
if(NOT DEFINED QUERY)
    message(STATUS "\t!!QUERY variable is NOT set, defaulting to QUERY=1!!")
    set(QUERY 1)
else()
    message(STATUS "\tQUERY=${QUERY}")
endif()

# Pick the default seed if the user did not specify one to CMake.
# We do a seed sweep to find a good seed by default
if(NOT DEFINED SEED)
    if(${QUERY} EQUAL 1)
        set(SEED 2)
    elseif(${QUERY} EQUAL 9)
        set(SEED 2)
    elseif(${QUERY} EQUAL 11)
        set(SEED 4)
    elseif(${QUERY} EQUAL 12)
        set(SEED 3)
    else()
        message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 9, 11 and 12)")
    endif()
endif()

# set number of CPU cores to use for quartus compile
if(NOT DEFINED QUARTUS_CORES)
    set(QUARTUS_CORES 2)
endif()

# check if they want to use the small database
if(DEFINED SF_SMALL)
    message(STATUS "\tManual override for database size - building for small database (SF=0.01)")
    set(SF_SMALL_ARG -DSF_SMALL)
else()
    set(SF_SMALL_ARG )
endif()

# setting source file based on query version
if(${QUERY} EQUAL 1)
    set(DEVICE_SOURCE query1/query1_kernel.cpp)
elseif(${QUERY} EQUAL 9)
    set(DEVICE_SOURCE query9/query9_kernel.cpp)
elseif(${QUERY} EQUAL 11)
    set(DEVICE_SOURCE query11/query11_kernel.cpp)
elseif(${QUERY} EQUAL 12)
    set(DEVICE_SOURCE query12/query12_kernel.cpp)
else()
    message(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 9, 11 and 12)")
endif()

# Check if target is the Intel(R) PAC D5005 (with Intel Stratix(R) 10 SX FPGA)
if(NOT DEFINED FPGA_BOARD OR FPGA_BOARD STREQUAL "intel_s10sx_pac:pac_s10")
    set(FPGA_BOARD "intel_s10sx_pac:pac_s10")
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) D5005 (with Intel Stratix(R) 10 SX FPGA).")
else()
    message(FATAL_ERROR "\tAn invalid board name was passed in using the FPGA_BOARD flag. Please refer to the README for the list of valid board names.")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=${QUERY} ${SF_SMALL_ARG}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -c -DQUERY=${QUERY} ${SF_SMALL_ARG}")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Wall -Xshardware -Xsboard=${FPGA_BOARD} -Xsopt-arg=\"-nocaching\" -Xsforce-single-store-ring -Xsparallel=${QUARTUS_CORES} -Xsseed=${SEED} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=1 query1_kernel.cpp db.cpp dbdata.cpp -o db.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=1 -o query1_kernel.cpp.o -c query1_kernel.cpp
#    [compile] dpcpp -fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=1 -o db.cpp.o -c db.cpp
#    [compile] dpcpp -fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=1 -o dbdata.cpp.o -c dbdata.cpp
#    [link]    dpcpp -fintelfpga query1_kernel.cpp.o db.cpp.o dbdata.cpp.o -o db.fpga_emu
add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE} ${HOST_SOURCE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -DQUERY=1 -Wall -Xshardware -Xsboard=<FPGA_BOARD> -Xsopt-arg="-nocaching" -Xsforce-single-store-ring -Xsparallel=2 -Xsseed=2 -DQUERY=1 -fsycl-link=early query1_kernel.cpp -o db_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${DEVICE_SOURCE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -DQUERY=1 -Wall -Xshardware -Xsboard=<FPGA_BOARD> -Xsopt-arg="-nocaching" -Xsforce-single-store-ring -Xsparallel=2 -Xsseed=2 -DQUERY=1 db.cpp dbdata.cpp query1_kernel.cpp -o db.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -DQUERY=1 -o db.cpp.o -c db.cpp
#   [compile] dpcpp -fintelfpga -DQUERY=1 -o dbdata.cpp.o -c dbdata.cpp
#   [compile] dpcpp -fintelfpga -DQUERY=1 -o query1_kernel.cpp.o -c query1_kernel.cpp
#   [link]    dpcpp -fintelfpga -Wall -Xshardware -Xsboard=<FPGA_BOARD> -Xsopt-arg="-nocaching" -Xsforce-single-store-ring -Xsparallel=2 -Xsseed=2 -DQUERY=1 db.cpp.o dbdata.cpp.o query1_kernel.cpp.o -o db.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${HOST_SOURCE} ${DEVICE_SOURCE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
