set(HOST_SOURCE db.cpp dbdata.cpp)
file(GLOB DB_UTILITY_HEADERS "db_util/*.hpp")
set(HOST_HEADER_FILES ${DB_UTILITY_HEADERS} dbdata.hpp)

set(TARGET_NAME db)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# set SEED variable
IF (NOT DEFINED SEED)
    SET(SEED 1)
ENDIF()

# set number of CPU cores to use for quartus compile
IF (NOT DEFINED QUARTUS_CORES)
    SET(QUARTUS_CORES 4)
ENDIF()

# check if they want to use the small database
IF (DEFINED SF_SMALL)
    MESSAGE(STATUS "\tManual override for database size - building for small database (SF=0.01)")
    SET(SF_SMALL_ARG -DSF_SMALL)
ELSE()
    SET(SF_SMALL_ARG )
ENDIF()

# which query are we doing?
IF (NOT DEFINED QUERY)
    MESSAGE(STATUS "\t!!QUERY variable is NOT set, defaulting to QUERY=1!!")
    SET(QUERY 1)
ELSE()
    MESSAGE(STATUS "\tQUERY=${QUERY}")
ENDIF()

# setting source file based on query version
IF(${QUERY} EQUAL 1)
    set(DEVICE_SOURCE query1/query1_kernel.cpp)
    set(DEVICE_HEADER query1/query1_kernel.hpp)
ELSEIF(${QUERY} EQUAL 9)
    set(DEVICE_SOURCE query9/query9_kernel.cpp)
    set(DEVICE_HEADER query9/query9_kernel.hpp)
ELSEIF(${QUERY} EQUAL 11)
    set(DEVICE_SOURCE query11/query11_kernel.cpp)
    set(DEVICE_HEADER query11/query11_kernel.hpp)
ELSEIF(${QUERY} EQUAL 12)
    set(DEVICE_SOURCE query12/query12_kernel.cpp)
    set(DEVICE_HEADER query12/query12_kernel.hpp)
ELSE()
    MESSAGE(FATAL_ERROR "\tQUERY ${QUERY} not supported (supported queries are 1, 9, 11 and 12)")
ENDIF()

# Intel supported FPGA Boards and their names
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")

# Assume target is the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA 
SET(_FPGA_BOARD ${S10_PAC_BOARD_NAME})

# Check if target is the Intel(R) PAC with Intel Stratix(R) 10 SX FPGA
IF (NOT DEFINED FPGA_BOARD)
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Stratix(R) 10 SX FPGA.")
ELSEIF(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    MESSAGE(FATAL_ERROR "\tThis example design ONLY supports the Intel(R) Programmable Acceleration Card (PAC) with Intel Stratix(R) 10 SX FPGA.")
ELSEIF(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    MESSAGE(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Stratix(R) 10 SX FPGA.")
ELSE()
    MESSAGE(FATAL_ERROR "\tAn invalid board name was passed in using the FPGA_BOARD flag. Please refer to the README for the list of valid board names.")
ENDIF()

set(HARDWARE_COMPILE_FLAGS "-fintelfpga -c -DQUERY=${QUERY} ${SF_SMALL_ARG}")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Wall -Xshardware -Xsboard=${_FPGA_BOARD} -Xsopt-arg=\"-nocaching\" -Xsforce-single-store-ring -Xsparallel=${QUARTUS_CORES} -Xsseed=${SEED} -DQUERY=${QUERY} ${SF_SMALL_ARG} ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -Wall -DFPGA_EMULATOR -DQUERY=${QUERY} ${SF_SMALL_ARG}")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

# fpga emulator
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE} ${HOST_SOURCE} ${HOST_HEADERS} ${DB_UTILITY_HEADERS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
endif()

# fpga
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${HOST_SOURCE} ${DEVICE_SOURCE})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
endif()

# fpga report
if(WIN32)
    message(FATAL_ERROR "No CMake support for Windows, yet")
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db.cpp db.cpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dbdata.cpp dbdata.cpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dbdata.hpp dbdata.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE} ${DEVICE_SOURCE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_HEADER} ${DEVICE_HEADER} COPYONLY)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/Accumulator.hpp db_utils/Accumulator.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/Date.hpp db_utils/Date.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/fifo_sort.hpp db_utils/fifo_sort.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/LikeRegex.hpp db_utils/LikeRegex.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/MapJoin.hpp db_utils/MapJoin.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/MergeJoin.hpp db_utils/MergeJoin.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/Misc.hpp db_utils/Misc.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/ShannonIterator.hpp db_utils/ShannonIterator.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/StreamingData.hpp db_utils/StreamingData.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/Tuple.hpp db_utils/Tuple.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/db_utils/Unroller.hpp db_utils/Unroller.hpp COPYONLY)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query1/query1_kernel.hpp query1/query1_kernel.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query9/query9_kernel.hpp query9/query9_kernel.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query9/pipe_types.hpp query9/pipe_types.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query11/query11_kernel.hpp query11/query11_kernel.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query11/pipe_types.hpp query11/pipe_types.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query12/query12_kernel.hpp query12/query12_kernel.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/query12/pipe_types.hpp query12/pipe_types.hpp COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${DEVICE_SOURCE} ${HOST_SOURCE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                 DEPENDS ${DEVICE_SOURCE} ${HOST_SOURCE} ${HOST_HEADERS} ${DB_UTILITY_HEADERS})

    add_custom_target(report DEPENDS ${REPORTS_TARGET})
endif()

