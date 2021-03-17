cmake_minimum_required (VERSION 3.5)
project (STREAM)

set(CMAKE_CXX_COMPILER dpcpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -DTUNED -DSTREAM_ARRAY_SIZE=134217728 -DNTIMES=20") 

add_executable(stream src/stream.cpp)

add_custom_target(run ./stream)
