if(WIN32)
  set(CMAKE_GENERATOR_PLATFORM
      x64
      CACHE STRING "")
  message(STATUS "Generator Platform set to ${CMAKE_GENERATOR_PLATFORM}")
endif()
