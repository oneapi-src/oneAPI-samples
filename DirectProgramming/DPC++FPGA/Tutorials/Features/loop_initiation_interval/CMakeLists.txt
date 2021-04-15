if(UNIX)
    # Direct CMake to use dpcpp rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER dpcpp)
else() # Windows
    # Force CMake to use dpcpp rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (dpcpp IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.4)

project(LoopInitiationInterval CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (src)
