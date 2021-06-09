# High Bandwidth variant source files
set(DEVICE_SOURCE_FILE gzipkernel.cpp)
set(DEVICE_HEADER_FILE gzipkernel.hpp) 
set(HOST_SOURCE_FILE gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)

# Low Latency variant source files
set(DEVICE_SOURCE_FILE_LL gzipkernel_ll.cpp)
set(DEVICE_HEADER_FILE_LL gzipkernel_ll.hpp)
set(HOST_SOURCE_FILE_LL gzip_ll.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)

set(TARGET_NAME gzip)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(EMULATOR_TARGET_LL ${TARGET_NAME}_ll.fpga_emu)

set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_LL ${TARGET_NAME}_ll.fpga)

set(REPORTS_TARGET ${TARGET_NAME}_report)
set(REPORTS_TARGET_LL ${TARGET_NAME}_ll_report)

set(HIGH_BW_SEED "-Xsseed=8")
set(LL_SEED "-Xsseed=1")

set (IS_A10 1)

# FPGA board selection
set(A10_PAC_BOARD_NAME "intel_a10gx_pac:pac_a10")
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")
set(S10_PAC_USM_BOARD_NAME "intel_s10sx_pac:pac_s10_usm")
set(SELECTED_BOARD ${A10_PAC_BOARD_NAME})
if (NOT DEFINED FPGA_BOARD)
    message(STATUS "\tFPGA_BOARD was not specified. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for information on board selection.")
    message(STATUS "\tNote: The Low-Latency Variant of the GZIP reference design not supported on Arria(R) 10 GX.")
elseif(FPGA_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA.")
    message(STATUS "\tNote: The Low-Latency Variant of the GZIP reference design not supported on Arria(R) 10 GX.")
elseif(FPGA_BOARD STREQUAL ${S10_PAC_BOARD_NAME})
    message(FATAL_ERROR "\tGZIP Reference design is not supported on pac_s10 board variant. Please use pac_s10_usm instead.")
elseif(FPGA_BOARD STREQUAL ${S10_PAC_USM_BOARD_NAME})
    message(STATUS "\tConfiguring the design to run on the Intel(R) Programmable Acceleration Card (PAC) D5005 (with Intel Stratix(R) 10 SX FPGA).")
    set(SELECTED_BOARD ${S10_PAC_USM_BOARD_NAME})
    set (IS_A10 0)
else()
    message(STATUS "\tAn invalid board name was passed in using the FPGA_BOARD flag. Configuring the design to run on the Intel(R) Programmable Acceleration Card (PAC) with Intel Arria(R) 10 GX FPGA. Please refer to the README for the list of valid board names.")
    message(STATUS "\tNote: The Low-Latency Variant of the GZIP reference design not supported on Arria(R) 10 GX.")
endif()

# Design specific constant values
# To increase NUM_ENGINES to greater than 2, must also statically declare more engines in gzipkernel.cpp --> SubmitGzipTasks()
if(SELECTED_BOARD STREQUAL ${A10_PAC_BOARD_NAME})
    # A10 parameters
    set(NUM_ENGINES 1)
    set(NUM_REORDER "")
elseif(SELECTED_BOARD STREQUAL ${S10_PAC_USM_BOARD_NAME})
    # S10 parameters
    set(NUM_ENGINES 2)
    # For the High Bandwidth variant, specify 6 reordering units to improve global memory read bandwidth across 4 channels of DDR.
    # For Low Latency variant this is not necessary since only one channel of global memory is used (host memory).
    set(NUM_REORDER "-Xsnum-reorder=6")
    set(NUM_REORDER_LL "")
else()
    message(FATAL_ERROR "Unknown board!")
endif()


# Compiler flags
set(EMULATOR_COMPILE_FLAGS "-g0 -fintelfpga  -DFPGA_EMULATOR -DNUM_ENGINES=${NUM_ENGINES}")
set(EMULATOR_LINK_FLAGS -fintelfpga)

# specify -MMD -fsycl-link-targets=... instead of -fintelfpga to workaround known issue; lower report quality 
set(HARDWARE_COMPILE_FLAGS -MMD -fintelfpga -c -DNUM_ENGINES=${NUM_ENGINES})

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsparallel=2 -Xsopt-arg="-nocaching" -Xsboard=${SELECTED_BOARD} ${USER_HARDWARE_FLAGS} -DNUM_ENGINES=${NUM_ENGINES})
set(FINAL_LINK_FLAGS -fintelfpga -DNUM_ENGINES=${NUM_ENGINES})


# FPGA emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp  -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})

    # Low-Latency Variant
    if (NOT ${IS_A10})
      set(WIN_EMULATOR_TARGET_LL ${EMULATOR_TARGET_LL}.exe)
      add_custom_target(fpga_emu_ll DEPENDS ${WIN_EMULATOR_TARGET_LL})
      separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
      add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET_LL} 
                         COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE_LL} ${CMAKE_CURRENT_SOURCE_DIR}/gzip_ll.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp  -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET_LL}
                         DEPENDS ${DEVICE_SOURCE_FILE_LL} ${HOST_SOURCE_FILE_LL})
    endif()
else()
    add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})

    # Low-Latency Variant
    if (NOT ${IS_A10})
      add_executable(${EMULATOR_TARGET_LL} ${DEVICE_SOURCE_FILE_LL} ${HOST_SOURCE_FILE_LL})
      add_custom_target(fpga_emu_ll DEPENDS ${EMULATOR_TARGET_LL})
      set_target_properties(${EMULATOR_TARGET_LL} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
      set_target_properties(${EMULATOR_TARGET_LL} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
    endif()

endif()


# FPGA hardware
if(WIN32)
    add_custom_target(fpga COMMAND echo "An FPGA hardware target is not provided on Windows. See README for details.")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "gzipkernel_fpga.o")
    set(DEVICE_IMAGE_FPGA_OBJ "gzipkernel_fpga.a")
    set(HOST_SOURCE_FILES_WITH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp)

    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")

    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})

    set(OBJ_FILES)
    foreach(HOST_FILE ${HOST_SOURCE_FILES_WITH_PATH})
      set(HOST_FPGA_OBJ ${HOST_FILE}.o)
      add_custom_command(OUTPUT ${HOST_FPGA_OBJ}
                         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${HOST_FILE} -o ${HOST_FPGA_OBJ}
                         DEPENDS ${HOST_FILE})
      list(APPEND OBJ_FILES ${HOST_FPGA_OBJ})
    endforeach()

    # High BW GZIP Variant
    add_custom_command(OUTPUT ${DEVICE_IMAGE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER} ${HIGH_BW_SEED} -fsycl-link=image ${DEVICE_FPGA_OBJ} -o ${DEVICE_IMAGE_FPGA_OBJ}
                       DEPENDS ${DEVICE_FPGA_OBJ} ${OBJ_FILES})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${FINAL_LINK_FLAGS} ${OBJ_FILES} ${DEVICE_IMAGE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                       DEPENDS ${DEVICE_IMAGE_FPGA_OBJ} ${OBJ_FILES})

    # Low Latency GZIP Variant
    if (NOT ${IS_A10})
      add_custom_target(fpga_ll DEPENDS ${FPGA_TARGET_LL})
      set(DEVICE_FPGA_OBJ_LL "gzipkernel_fpga_ll.o")
      set(DEVICE_IMAGE_FPGA_OBJ_LL "gzipkernel_fpga_ll.a")
      set(HOST_SOURCE_FILES_WITH_PATH_LL ${CMAKE_CURRENT_SOURCE_DIR}/gzip_ll.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp)

      add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ_LL}
                         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE_LL} -o ${DEVICE_FPGA_OBJ_LL}
                         DEPENDS ${DEVICE_SOURCE_FILE_LL} ${DEVICE_HEADER_FILE_LL})

      set(OBJ_FILES_LL)
      foreach(HOST_FILE ${HOST_SOURCE_FILES_WITH_PATH_LL})
        set(HOST_FPGA_OBJ_LL ${HOST_FILE}.o)
        add_custom_command(OUTPUT ${HOST_FPGA_OBJ_LL}
                           COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_COMPILE_FLAGS} ${HOST_FILE} -o ${HOST_FPGA_OBJ_LL}
                           DEPENDS ${HOST_FILE})
        list(APPEND OBJ_FILES_LL ${HOST_FPGA_OBJ_LL})
      endforeach()

      add_custom_command(OUTPUT ${DEVICE_IMAGE_FPGA_OBJ_LL}
                         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER_LL} ${LL_SEED} -fsycl-link=image ${DEVICE_FPGA_OBJ_LL} -o ${DEVICE_IMAGE_FPGA_OBJ_LL}
                         DEPENDS ${DEVICE_FPGA_OBJ_LL} ${OBJ_FILES_LL})

      add_custom_command(OUTPUT ${FPGA_TARGET_LL}
                         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${FINAL_LINK_FLAGS} ${OBJ_FILES_LL} ${DEVICE_IMAGE_FPGA_OBJ_LL} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET_LL}
                         DEPENDS ${DEVICE_IMAGE_FPGA_OBJ_LL} ${OBJ_FILES_LL})
    endif()
endif()

# FPGA hardware report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )
    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER} -fsycl-link  ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})

    # Low-Latency Variant
    if (NOT ${IS_A10})
      add_custom_target(report_ll DEPENDS ${REPORTS_TARGET_LL} )
      separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
      add_custom_command(OUTPUT ${REPORTS_TARGET_LL}
                        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER_LL} -fsycl-link  ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE_LL} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET_LL}
                        DEPENDS ${DEVICE_SOURCE_FILE_LL} ${DEVICE_HEADER_FILE_LL})
    endif()
else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )
    

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${DEVICE_SOURCE_FILE} COPYONLY)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_HEADER_FILE} ${DEVICE_HEADER_FILE} COPYONLY)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.hpp CompareGzip.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/crc32.hpp crc32.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gzipkernel.hpp gzipkernel.hpp COPYONLY)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernels.hpp kernels.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.hpp WriteGzip.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_array.hpp pipe_array.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_array_internal.hpp pipe_array_internal.hpp COPYONLY)

    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")

    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER} -fsycl-link ${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE} kernels.hpp)

    # Low-Latency Variant
    if (NOT ${IS_A10})
      add_custom_target(report_ll DEPENDS ${REPORTS_TARGET_LL} )
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE_LL} ${DEVICE_SOURCE_FILE_LL} COPYONLY)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_HEADER_FILE_LL} ${DEVICE_HEADER_FILE_LL} COPYONLY)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gzipkernel_ll.hpp gzipkernel_ll.hpp COPYONLY)

      add_custom_command(OUTPUT ${REPORTS_TARGET_LL}
                        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS} ${NUM_REORDER_LL} -fsycl-link ${DEVICE_SOURCE_FILE_LL} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET_LL}
                        DEPENDS ${DEVICE_SOURCE_FILE_LL} ${DEVICE_HEADER_FILE_LL} kernels.hpp)
    endif()
endif()

