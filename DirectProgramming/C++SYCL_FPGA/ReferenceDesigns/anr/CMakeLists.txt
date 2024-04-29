# Direct CMake to use icpx rather than the default C++ compiler/linker on Linux
# and icx-cl on Windows
if(UNIX)
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.7.2)

project(anr CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILES src/main.cpp)
set(TARGET_NAME anr)

# Use cmake -DFPGA_DEVICE=<board-support-package>:<board-variant> to choose a
# different device.
# Note that depending on your installation, you may need to specify the full 
# path to the board support package (BSP), this usually is in your install 
# folder.
#
# You can also specify a device family (E.g. "Arria10" or "Stratix10") or a
# specific part number (E.g. "10AS066N3F40E2SG") to generate a standalone IP.
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to the default FPGA family target: ${FPGA_DEVICE}\
                    \nPlease refer to the README for information on target selection.")
    set(DEVICE_FLAG "Agilex7")
else()
    message(STATUS "Configuring the design with the following target: ${FPGA_DEVICE}")

    string(TOLOWER ${FPGA_DEVICE} FPGA_DEVICE_NAME)
    if(FPGA_DEVICE_NAME MATCHES ".*a10.*" OR FPGA_DEVICE_NAME MATCHES ".*arria10.*")
      set(DEVICE_FLAG "A10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*s10.*" OR FPGA_DEVICE_NAME MATCHES ".*stratix10.*")
      set(DEVICE_FLAG "S10")
    elseif(FPGA_DEVICE_NAME MATCHES ".*agilex.*")
      set(DEVICE_FLAG "Agilex7")
    endif()

    # Check if the target is a BSP
    if(IS_BSP MATCHES "1")
        set(BSP_FLAG "-DIS_BSP")
    else()
        message(STATUS "The selected target ${FPGA_DEVICE} is assumed to be an FPGA part number, so USM will be enabled by default.")
        message(STATUS "If the target is actually a BSP that does not support USM, run cmake with -DIS_BSP=1.")
    endif()
endif()

if(NOT DEFINED DEVICE_FLAG)
    message(FATAL_ERROR "An unrecognized or custom board was passed, but DEVICE_FLAG was not specified. \
                         Please make sure you have set -DDEVICE_FLAG=A10, -DDEVICE_FLAG=S10 or \
                         -DDEVICE_FLAG=Agilex7.")
endif()

# Increase the allowable constexpr steps for the front end. This allows the
#  front-end compiler to do more compile-time computation.
if(WIN32)
    set(CONSTEXPR_STEPS "-constexpr:steps5084968")
else()
    set(CONSTEXPR_STEPS "-fconstexpr-steps=5084968")
endif()

# Choose the random seed for the hardware compile
# e.g. cmake .. -DSEED=7
if(NOT DEFINED SEED)
    # the default seed
  if(DEVICE_FLAG MATCHES "A10")
    set(SEED 1)
  elseif(DEVICE_FLAG MATCHES "S10")
    set(SEED 2)
  elseif(DEVICE_FLAG MATCHES "Agilex7")
    set(SEED 3)
  else()
    set(SEED 4)
  endif()
else()
    message(STATUS "Seed explicitly set to ${SEED}")
endif()

if(IGNORE_DEFAULT_SEED)
    set(SEED_FLAG "")
else()
    set(SEED_FLAG "-Xsseed=${SEED}")
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
  if(DEVICE_FLAG MATCHES "A10")
    set(PIXELS_PER_CYCLE 2)
  elseif(DEVICE_FLAG MATCHES "S10")
    set(PIXELS_PER_CYCLE 2)
  elseif(DEVICE_FLAG MATCHES "Agilex7")
    set(PIXELS_PER_CYCLE 1)
  else()
    message(WARNING "Unknown board: setting PIXELS_PER_CYCLE to 1")
    set(PIXELS_PER_CYCLE 1)
  endif()
endif()
set(PIXELS_PER_CYCLE_FLAG "-DPIXELS_PER_CYCLE=${PIXELS_PER_CYCLE}")

# Allow the user to change the maximum number of pixels per column
# e.g. cmake .. -DMAX_COLS=3840
if(MAX_COLS)
    set(MAX_COLS_FLAG "-DMAX_COLS=${MAX_COLS}")
    message(STATUS "MAX_COLS explicitly set to ${MAX_COLS}")
endif()

# Allow the user to change the bitwidth of the pixels
# e.g. cmake .. -DPIXEL_BITS=8
if(PIXEL_BITS)
    set(PIXEL_BITS_FLAG "-DPIXEL_BITS=${PIXEL_BITS}")
    message(STATUS "PIXEL_BITS explicitly set to ${PIXEL_BITS}")
endif()

# Print out configured variables
message(STATUS "  SEED=${SEED_FLAG}")
message(STATUS "  PIXELS_PER_CYCLE=${PIXELS_PER_CYCLE}")
if(FILTER_SIZE)
  message(STATUS "  FILTER_SIZE=${FILTER_SIZE}")
endif()
if(PIXEL_BITS)
  message(STATUS "  PIXEL_BITS=${PIXEL_BITS}")
endif()

# Use cmake -DUSER_FPGA_FLAGS=<flags> to set extra flags for FPGA backend
# compilation. 
set(USER_FPGA_FLAGS ${USER_FPGA_FLAGS};${SEED_FLAG})

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS ${USER_FLAGS};${CONSTEXPR_STEPS};${FILTER_SIZE_FLAG};${PIXELS_PER_CYCLE_FLAG};${MAX_COLS_FLAG};${PIXEL_BITS_FLAG};${BSP_FLAG})

# Use cmake -DUSER_INCLUDE_PATHS=<paths> to set extra paths for general
# compilation.
set(USER_INCLUDE_PATHS ../../include;${USER_INCLUDE_PATHS})

###############################################################################
### no changes after here
###############################################################################

# Set the names of the makefile targets to be generated by cmake
set(EMULATOR_TARGET fpga_emu)
set(SIMULATOR_TARGET fpga_sim)
set(REPORT_TARGET report)
set(FPGA_TARGET fpga)

# Set the names of the generated files per makefile target
set(EMULATOR_OUTPUT_NAME ${TARGET_NAME}.${EMULATOR_TARGET})
set(SIMULATOR_OUTPUT_NAME ${TARGET_NAME}.${SIMULATOR_TARGET})
set(REPORT_OUTPUT_NAME ${TARGET_NAME}.${REPORT_TARGET})
set(FPGA_OUTPUT_NAME ${TARGET_NAME}.${FPGA_TARGET})

message(STATUS "Additional USER_FPGA_FLAGS=${USER_FPGA_FLAGS}")
message(STATUS "Additional USER_FLAGS=${USER_FLAGS}")

include_directories(${USER_INCLUDE_PATHS})
message(STATUS "Additional USER_INCLUDE_PATHS=${USER_INCLUDE_PATHS}")

link_directories(${USER_LIB_PATHS})
message(STATUS "Additional USER_LIB_PATHS=${USER_LIB_PATHS}")

link_libraries(${USER_LIBS})
message(STATUS "Additional USER_LIBS=${USER_LIBS}")

if(WIN32)
    # add qactypes for Windows
    set(QACTYPES "-Qactypes")
    # This is a Windows-specific flag that enables exception handling in host code
    set(WIN_FLAG "/EHsc")
else()
    # add qactypes for Linux
    set(QACTYPES "-qactypes")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" LOWER_BUILD_TYPE)
if(LOWER_BUILD_TYPE MATCHES debug)
# Set debug flags
    if(WIN32)
        set(DEBUG_FLAGS /DEBUG /Od)
    else()
        set(DEBUG_FLAGS -g -O0)
    endif()
else()
    set(DEBUG_FLAGS "")
endif()

if(WIN32)
    set(EXT ".exe")
endif()

set(COMMON_COMPILE_FLAGS -fintelfpga -Wall ${WIN_FLAG} ${QACTYPES} ${USER_FLAGS})
set(COMMON_LINK_FLAGS -fintelfpga ${QACTYPES} ${USER_FLAGS})

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate
#    representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking. For
#    this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS -DFPGA_EMULATOR ${DEBUG_FLAGS})
set(EMULATOR_LINK_FLAGS )
set(REPORT_COMPILE_FLAGS -DFPGA_HARDWARE)
set(REPORT_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -fsycl-link=early)
set(SIMULATOR_COMPILE_FLAGS -Xssimulation -DFPGA_SIMULATOR)
set(SIMULATOR_LINK_FLAGS -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_OUTPUT_NAME}${EXT})
set(FPGA_COMPILE_FLAGS -DFPGA_HARDWARE)
set(FPGA_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_OUTPUT_NAME}${EXT})

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILES})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${EMULATOR_COMPILE_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${EMULATOR_LINK_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${EMULATOR_OUTPUT_NAME})

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILES})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${SIMULATOR_COMPILE_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${SIMULATOR_LINK_FLAGS})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${SIMULATOR_OUTPUT_NAME})

###############################################################################
### Generate Report
###############################################################################
add_executable(${REPORT_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILES})
target_compile_options(${REPORT_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${REPORT_TARGET} PRIVATE ${REPORT_COMPILE_FLAGS})

# The report target does not need the QACTYPES flag at link stage
set(MODIFIED_COMMON_LINK_FLAGS_REPORT ${COMMON_LINK_FLAGS})
list(REMOVE_ITEM MODIFIED_COMMON_LINK_FLAGS_REPORT ${QACTYPES})

target_link_libraries(${REPORT_TARGET} ${MODIFIED_COMMON_LINK_FLAGS_REPORT})
target_link_libraries(${REPORT_TARGET} ${REPORT_LINK_FLAGS})
set_target_properties(${REPORT_TARGET} PROPERTIES OUTPUT_NAME ${REPORT_OUTPUT_NAME})

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILES})
target_compile_options(${FPGA_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${FPGA_TARGET} PRIVATE ${FPGA_COMPILE_FLAGS})
target_link_libraries(${FPGA_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${FPGA_TARGET} ${FPGA_LINK_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES OUTPUT_NAME ${FPGA_OUTPUT_NAME})

###############################################################################
### This part only manipulates cmake variables to print the commands cmake is expected to run to the user
###############################################################################

# set the correct object file extension depending on the target platform
if(WIN32)
    set(OBJ_EXTENSION "obj")
else()
    set(OBJ_EXTENSION "o")
endif()

# Set the source file names in a string
set(SOURCE_FILE_NAME "${SOURCE_FILES}")

function(getCompileCommands common_compile_flags special_compile_flags common_link_flags special_link_flags target output_name)

    set(file_names ${SOURCE_FILE_NAME})
    set(COMPILE_COMMAND )
    set(LINK_COMMAND )

    foreach(source ${file_names})
        # Get the relative path to the source and object files
        file(RELATIVE_PATH CURRENT_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${source})
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})
        
        # Creating a string that contains the compile command
        # Start by the compiler invocation
        set(COMPILE_COMMAND "${COMPILE_COMMAND}${CMAKE_CXX_COMPILER}")

        # Add all the potential includes
        foreach(INCLUDE ${USER_INCLUDE_PATHS})
            if(NOT IS_ABSOLUTE ${INCLUDE})
                file(RELATIVE_PATH INCLUDE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${INCLUDE})
            endif()
            set(COMPILE_COMMAND "${COMPILE_COMMAND} -I${INCLUDE}")
        endforeach()

        # Add all the common compile flags
        foreach(FLAG ${common_compile_flags})
            set(COMPILE_COMMAND "${COMPILE_COMMAND} ${FLAG}")
        endforeach()

        # Add all the specific compile flags
        foreach(FLAG ${special_compile_flags})
            set(COMPILE_COMMAND "${COMPILE_COMMAND} ${FLAG}")
        endforeach()

        # Get the location of the object file
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})

        # Add the source file and the output file
        set(COMPILE_COMMAND "${COMPILE_COMMAND} -c ${CURRENT_SOURCE_FILE} -o ${OBJ_FILE}\n")
    endforeach()

    set(COMPILE_COMMAND "${COMPILE_COMMAND}" PARENT_SCOPE)

    # Creating a string that contains the link command
    # Start by the compiler invocation
    set(LINK_COMMAND "${LINK_COMMAND}${CMAKE_CXX_COMPILER}")

    # Add all the common link flags
    foreach(FLAG ${common_link_flags})
        set(LINK_COMMAND "${LINK_COMMAND} ${FLAG}")
    endforeach()

    # Add all the specific link flags
    foreach(FLAG ${special_link_flags})
        set(LINK_COMMAND "${LINK_COMMAND} ${FLAG}")
    endforeach()    

    # Add the output file
    set(LINK_COMMAND "${LINK_COMMAND} -o ${output_name}")

    foreach(source ${file_names})
        # Get the relative path to the source and object files
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})

        # Add the source file and the output file
        set(LINK_COMMAND "${LINK_COMMAND} ${OBJ_FILE}")
    endforeach()

    # Add all the potential library paths
    foreach(LIB_PATH ${USER_LIB_PATHS})
        if(NOT IS_ABSOLUTE ${LIB_PATH})
            file(RELATIVE_PATH LIB_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${LIB_PATH})
        endif()
        if(NOT WIN32)
            set(LINK_COMMAND "${LINK_COMMAND} -L${LIB_PATH}")
        else()
            set(LINK_COMMAND "${LINK_COMMAND} -L${LIB_PATH} -Wl,-rpath,${LIB_PATH}")
        endif()
    endforeach()

    # Add all the potential includes
    foreach(LIB ${USER_LIBS})
        set(LINK_COMMAND "${LINK_COMMAND} -l${LIB}")
    endforeach()

    set(LINK_COMMAND "${LINK_COMMAND}" PARENT_SCOPE)

endfunction()

# Windows executable is going to have the .exe extension
if(WIN32)
    set(EXECUTABLE_EXTENSION ".exe")
endif()

# Display the compile instructions in the emulation flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${EMULATOR_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${EMULATOR_LINK_FLAGS}" "${EMULATOR_TARGET}" "${EMULATOR_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayEmulationCompileCommands
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "\nTo compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${EMULATOR_TARGET} displayEmulationCompileCommands)

# Display the compile instructions in the simulation flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${SIMULATOR_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${SIMULATOR_LINK_FLAGS}" "${SIMULATOR_TARGET}" "${SIMULATOR_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displaySimulationCompileCommands
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "\nTo compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${SIMULATOR_TARGET} displaySimulationCompileCommands)

# Display the compile instructions in the report flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${REPORT_COMPILE_FLAGS}" "${MODIFIED_COMMON_LINK_FLAGS_REPORT}" "${REPORT_LINK_FLAGS}" "${REPORT_TARGET}" "${REPORT_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayReportCompileCommands
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "\nTo compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${REPORT_TARGET} displayReportCompileCommands)

# Display the compile instructions in the fpga flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${FPGA_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${FPGA_LINK_FLAGS}" "${FPGA_TARGET}" "${FPGA_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayFPGACompileCommands
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "\nTo compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${FPGA_TARGET} displayFPGACompileCommands)
