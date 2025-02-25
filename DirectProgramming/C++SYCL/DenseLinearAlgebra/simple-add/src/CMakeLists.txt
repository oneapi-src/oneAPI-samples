# Select between the USM and buffer variant of the
# code to compile, depending on the value of USM
# given to cmake
# e.g. if cmake is called with -DUSM=1, the USM
# source code will be compiled
if(DEFINED USM AND (NOT(USM EQUAL 0)))
    message(STATUS "Using the USM variant.")
    set(SOURCE_FILE simple-add-usm.cpp)
    set(TARGET_NAME simple-add-usm)
else()
    set(SOURCE_FILE simple-add-buffers.cpp)
    set(TARGET_NAME simple-add-buffers)
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})
