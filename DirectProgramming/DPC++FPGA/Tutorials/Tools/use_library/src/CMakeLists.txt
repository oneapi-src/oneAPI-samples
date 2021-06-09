set(SOURCE_FILE use_library.cpp)
set(HEADER_FILE lib.hpp)
set(LIBRARY_ARCHIVE lib.a)
set(TARGET_NAME use_library)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORT_TARGET ${TARGET_NAME}_report.a)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

# Library source files
set(HLS_SOURCE lib_hls.cpp)
set(HLS_SOURCE_OBJECT lib_hls.o)
set(OCL_SOURCE lib_ocl.cl)
set(OCL_SOURCE_OBJECT lib_ocl.o)
set(SYCL_SOURCE lib_sycl.cpp)
set(SYCL_SOURCE_OBJECT lib_sycl.o)
set(RTL_C_MODEL lib_rtl_model.cpp)
set(RTL_SPEC lib_rtl_spec.xml)
set(RTL_V lib_rtl.v)
set(RTL_SOURCE_OBJECT lib_rtl.o)

# Flags
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

# Flag lists for custom commands
separate_arguments(CMAKE_CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")

###############################################################################
### Generate Library
###############################################################################

# The RTL file (specified in lib_rtl_spec.xml) must be copied to the CMake working directory for the final stage of FPGA hardware compilation
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${RTL_V} ${RTL_V} COPYONLY)

# Create HLS source object
add_custom_target(
    create_hls_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${HLS_SOURCE} --source hls --target sycl -o ${HLS_SOURCE_OBJECT} ${CMAKE_CXX_FLAGS_LIST}
)

# Create OCL source object
add_custom_target(
    create_ocl_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${OCL_SOURCE} --source ocl --target sycl -o ${OCL_SOURCE_OBJECT}
)

# Create SYCL source object
add_custom_target(
    create_sycl_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${SYCL_SOURCE} --source sycl --target sycl -o ${SYCL_SOURCE_OBJECT} ${CMAKE_CXX_FLAGS_LIST}
)

# Create RTL source object
add_custom_target(
    create_rtl_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${RTL_SPEC} --emulation_model ${CMAKE_CURRENT_SOURCE_DIR}/${RTL_C_MODEL} --target sycl -o ${RTL_SOURCE_OBJECT}
)

# Create library archive
# This executes:
# fpga_libtool lib_hls.o lib_ocl.o lib_sycl.o lib_rtl.o --target sycl --create lib.a
add_custom_target(create_library_archive DEPENDS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE})
add_custom_command(OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   COMMAND fpga_libtool ${HLS_SOURCE_OBJECT} ${OCL_SOURCE_OBJECT} ${SYCL_SOURCE_OBJECT} ${RTL_SOURCE_OBJECT} --target sycl --create ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   DEPENDS create_hls_source_object create_ocl_source_object create_sycl_source_object create_rtl_source_object)

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
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o use_library.cpp.o -c use_library.cpp
#    [link]    dpcpp -fintelfpga use_library.cpp.o -o use_library.fpga_emu lib.a 
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
target_link_libraries(${EMULATOR_TARGET} library_archive)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early use_library.cpp lib.a -o use_library_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_link_libraries(${FPGA_EARLY_IMAGE} library_archive)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> use_library.cpp lib.a -o use_library.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o use_library.cpp.o -c use_library.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> use_library.cpp.o -o use_library.fpga lib.a 
add_executable(${FPGA_TARGET} ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
target_link_libraries(${FPGA_TARGET} library_archive)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
