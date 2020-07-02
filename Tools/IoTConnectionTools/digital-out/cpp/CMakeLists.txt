set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MRAA_CFLAGS}")
add_executable (digital-out digital-out.cpp)
add_custom_target (run ./digital-out)
target_link_libraries(digital-out ${MRAA_LDFLAGS})
