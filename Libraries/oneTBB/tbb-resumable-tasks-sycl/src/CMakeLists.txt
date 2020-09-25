set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_COMPILER "${CMAKE_C_COMPILER}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -cl-opt-disable")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -ltbb")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

add_executable (tbb-resumable-tasks-sycl tbb-resumable-tasks-sycl.cpp)

find_package(Threads REQUIRED)
target_link_libraries(tbb-resumable-tasks-sycl Threads::Threads)

add_custom_target (run ./tbb-resumable-tasks-sycl)
