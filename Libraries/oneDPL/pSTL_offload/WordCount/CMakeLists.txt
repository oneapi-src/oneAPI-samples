if(GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-pstl-offload=gpu -D GPU")
endif()
if(CPU)
  #To build for CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-ltbb ${CMAKE_EXE_LINKER_FLAGS}")

add_executable (WordCount main.cpp)

add_custom_target (run_wc cd ${CMAKE_SOURCE_DIR}/WordCount/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/WordCount)