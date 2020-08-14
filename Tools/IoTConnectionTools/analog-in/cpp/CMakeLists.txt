set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MRAA_CFLAGS}")
add_executable (analog-in analog-in.cpp)
add_custom_target (run ./analog-in)
target_link_libraries(analog-in ${MRAA_LDFLAGS})
