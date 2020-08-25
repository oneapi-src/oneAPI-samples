if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Add an executable target from source files
add_executable(${PROJECT_NAME} main.cpp)

# Specify compilation flags for the target
target_compile_options(${PROJECT_NAME} PRIVATE -fsycl -fsycl-unnamed-lambda)

# Specify compilation definitions for the target
target_compile_definitions(${PROJECT_NAME} PRIVATE -D_PSTL_BACKEND_SYCL)

# Specify libraries to link with
target_link_libraries(${PROJECT_NAME} OpenCL sycl)

# Add custom target for running
add_custom_target(run ./${PROJECT_NAME})
