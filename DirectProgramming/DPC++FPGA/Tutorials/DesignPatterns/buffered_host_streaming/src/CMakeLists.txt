set(SOURCE_FILE buffered_host_streaming.cpp)
set(TARGET_NAME buffered_host_streaming)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

# USM is only supported on the PAC S10 board
set(S10_PAC_USM_BOARD_NAME "intel_s10sx_pac:pac_s10_usm")

# Flags
if(WIN32)
    set(THREAD_FLAG "")
else()
    set(THREAD_FLAG "-lpthread")
endif()

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -Wall -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${THREAD_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga -c")
set(HARDWARE_LINK_FLAGS "-fintelfpga ${THREAD_FLAG} -Wall -Xshardware -Xsboard=${S10_PAC_USM_BOARD_NAME} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation


# FPGA emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                       DEPENDS ${SOURCE_FILE})
else()
    add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# FPGA hardware
if(WIN32)
    add_custom_target(fpga COMMAND echo "An FPGA hardware target is not provided on Windows. See README for details.")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# FPGA hardware report
if(WIN32)
    set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
    add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})
    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_OBJ_FILE} 
                       COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                       DEPENDS ${SOURCE_FILE})
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/common.hpp common.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HostStreamer.hpp HostStreamer.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_api.hpp streaming_with_api.hpp COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/streaming_without_api.hpp streaming_without_api.hpp COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    add_custom_command(OUTPUT ${REPORTS_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_LIST} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                       DEPENDS ${SOURCE_FILE})
    add_custom_target(report DEPENDS ${REPORTS_TARGET})
endif()

