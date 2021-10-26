add_library(utils STATIC
  image_io.h
  image_io.cpp
)

target_link_libraries(utils PUBLIC common)
