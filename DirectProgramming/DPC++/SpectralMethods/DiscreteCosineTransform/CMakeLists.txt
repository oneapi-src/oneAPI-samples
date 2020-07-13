if(WIN32)
        set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
        set(CMAKE_CXX_COMPILER "dpcpp")
endif()
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
cmake_minimum_required (VERSION 3.0)
project (DiscreteCosineTransform)
add_subdirectory (src)
