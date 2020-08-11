set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MRAA_CFLAGS}")
add_executable (up2-leds up2-leds.c)
add_custom_target (run ./up2-leds)
target_link_libraries(up2-leds ${MRAA_LDFLAGS})
