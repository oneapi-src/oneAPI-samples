add_library(utils STATIC
  arg_parser.h
  arg_parser.cpp
  device_info.h
  image_buffer.h
  image_buffer.cpp
  image_io.h
  image_io.cpp
  random.h
)

target_link_libraries(utils PUBLIC common OpenImageDenoise)
