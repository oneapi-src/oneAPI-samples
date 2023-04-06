project(test_HEAAN)
# original makefile
# g++ test.cpp -o TestHEAAN -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread

find_package (Threads REQUIRED)

# SOME DEBUG CODE
#find_library(lib_ntl_location NAMES libntl.a)
#message(STATUS ${lib_ntl_location})
#/usr/local/lib is not one of the implicit link directories. Therefore, a quick fix is to include this line in the config
#set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES /usr/local/lib ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})


set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (TEST_SOURCES
    test.cpp
    )

set (TEST_COMPILE_OPTS
    "-O2;"
    )

#
# WARNING: linking ORDER matters here
#
set (TEST_LINK_OPTS
    "-lntl;-lgmp;-lm"
    )

add_executable (TestHEAAN
    ${TEST_SOURCES}
    )

include_directories (${PROJECT_SOURCE_DIR}/../src)

target_compile_options (
    TestHEAAN
    PRIVATE  ${TEST_COMPILE_OPTS}
    )

target_link_libraries(
    TestHEAAN
    PRIVATE  ${TEST_LINK_OPTS}
    Threads::Threads
    PRIVATE HEAAN_lib
    )
