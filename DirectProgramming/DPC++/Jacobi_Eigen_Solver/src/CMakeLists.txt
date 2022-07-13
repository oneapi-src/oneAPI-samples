set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -O1 -llapack")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(jacobi_solver jacobi_solver.cpp)
target_link_libraries(jacobi_solver OpenCL sycl)

option(GPU_SELECTOR "Option description" OFF)
option(CPU_SELECTOR "Option description" OFF)

if(GPU_SELECTOR)
  add_definitions(-D_GPU)
elseif(CPU_SELECTOR)
  add_definitions(-D_CPU)
endif()

add_custom_target(run ./jacobi_solver)
add_custom_target(gpu )
