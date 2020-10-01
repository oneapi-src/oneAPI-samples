set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MRAA_CFLAGS}")
add_executable (interrupt interrupt.cpp)
add_custom_target (run ./interrupt)
target_link_libraries(interrupt ${MRAA_LDFLAGS})
