CMAKE_MINIMUM_REQUIRED (VERSION 3.13)
project (HEAAN LANGUAGES CXX)

include (CheckLanguage)
check_language (CXX)

if (CMAKE_CXX_COMPILER)
  enable_language (CXX)
endif()

#
# debug cmake and make: turn ON for debug
#
set (CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set (CMAKE_VERBOSE_MAKEFILE OFF)

add_subdirectory (HEAAN)
