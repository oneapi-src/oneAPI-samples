set(SOURCE_FILE use_library.cpp)
set(HEADER_FILE lib.hpp)
set(LIBRARY_ARCHIVE lib.a)
set(TARGET_NAME use_library)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORT_TARGET ${TARGET_NAME}_report.a)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

# Library source files
set(RTL_C_MODEL lib_rtl_model.cpp)
set(RTL_SPEC lib_rtl_spec.xml)
set(RTL_V lib_rtl.v)
set(RTL_SOURCE_OBJECT lib_rtl.o)

# Flags
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(SIMULATOR_COMPILE_FLAGS "-fintelfpga")
set(SIMULATOR_LINK_FLAGS "-fintelfpga -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_SIMULATOR_FLAGS} -DFPGA_SIMULATOR")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

# Flag lists for custom commands
separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")

###############################################################################
### Generate Library
###############################################################################

# The RTL file (specified in lib_rtl_spec.xml) must be copied to the CMake working directory for the final stage of FPGA hardware compilation
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${RTL_V} ${RTL_V} COPYONLY)

# Create RTL source object
add_custom_target(
    create_rtl_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${RTL_SPEC} --emulation_model ${CMAKE_CURRENT_SOURCE_DIR}/${RTL_C_MODEL} --target sycl -o ${RTL_SOURCE_OBJECT}
)

# Create library archive
# This executes:
# fpga_libtool lib_rtl.o --target sycl --create lib.a
add_custom_target(create_library_archive DEPENDS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE})
add_custom_command(OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   COMMAND fpga_libtool ${RTL_SOURCE_OBJECT} --target sycl --create ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   DEPENDS create_rtl_source_object)

# Tell CMake to recognize our custom library
add_library(library_archive STATIC IMPORTED GLOBAL)
add_dependencies(library_archive create_library_archive)
set_target_properties(library_archive PROPERTIES IMPORTED_LOCATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE})

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR use_library.cpp lib.a -o use_library.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o use_library.o -c use_library.cpp
#    [link]    dpcpp -fintelfpga use_library.cpp.o -o use_library.fpga_emu lib.a lib_sycl.o
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
target_link_libraries(${EMULATOR_TARGET} library_archive)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early use_library.cpp lib.a -o use_library_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_link_libraries(${FPGA_EARLY_IMAGE} library_archive)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xssimulation -Xsghdl -DFPGA_SIMULATOR -Xstarget=<FPGA_DEVICE> use_library.cpp lib.a -o use_library.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o use_library.cpp.o -c use_library.cpp -DFPGA_SIMULATOR
#   [link]    dpcpp -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> use_library.cpp.o -o use_library.fpga lib.a  lib_sycl.o
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${SIMULATOR_TARGET})
target_link_libraries(${SIMULATOR_TARGET} library_archive)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${SIMULATOR_COMPILE_FLAGS}")
set_target_properties(${SIMULATOR_TARGET} PROPERTIES LINK_FLAGS "${SIMULATOR_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_TARGET}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> use_library.cpp lib.a -o use_library.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o use_library.cpp.o -c use_library.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> use_library.cpp.o -o use_library.fpga lib.a  lib_sycl.o
add_executable(${FPGA_TARGET} ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
target_link_libraries(${FPGA_TARGET} library_archive)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
