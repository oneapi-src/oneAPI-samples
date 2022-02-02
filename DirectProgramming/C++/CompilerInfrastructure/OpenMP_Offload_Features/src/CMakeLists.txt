if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fiopenmp -fopenmp-targets=spir64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Add an executable target from source files
add_executable(prog1 class_member_functor.cpp)
add_custom_target(run_prog1 ./prog1)

add_executable(prog2 function_pointer.cpp)
add_custom_target(run_prog2 ./prog2)

add_executable(prog3 user_defined_mapper.cpp)
add_custom_target(run_prog3 ./prog3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fiopenmp -fopenmp-targets=spir64 -fsycl")
add_executable(prog4 usm_and_composability_with_dpcpp.cpp)
target_link_libraries(prog4 sycl)
add_custom_target(run_prog4 ./prog4)
