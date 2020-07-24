if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Add an executable target from source files
add_executable(${PROJECT_NAME} main.cpp)

# Add custom target for running
add_custom_target(run ./${PROJECT_NAME})
