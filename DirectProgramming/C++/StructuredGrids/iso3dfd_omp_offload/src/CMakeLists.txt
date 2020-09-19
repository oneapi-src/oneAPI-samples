OPTION(VERIFY_RESULTS "Use Results Validation" ON)
OPTION(USE_OPT1 "Select Optimized target code - version 1" OFF)
OPTION(USE_OPT2 "Select Optimized target code - version 2" OFF)
OPTION(USE_OPT3 "Select Optimized target code - version 3" OFF)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fiopenmp -std=c++17 -fopenmp-targets=spir64 -O3 -D__STRICT_ANSI__ ")

set(SOURCES iso3dfd.cpp utils.cpp)

if(USE_OPT3)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPT3")
	message("-- Using Optimized target code - version 3")
elseif(USE_OPT2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPT2")
	message("-- Using Optimized target code - version 2")
elseif(USE_OPT1)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPT1")
	message("-- Using Optimized target code - version 1")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BASELINE")
	message("-- Using Baseline target code")
endif()

if(VERIFY_RESULTS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERIFY_RESULTS")
	set(SOURCES ${SOURCES} iso3dfd_verify.cpp)
endif(VERIFY_RESULTS)


add_executable (iso3dfd ${SOURCES})

add_custom_target (run 
	COMMAND iso3dfd 256 256 256 16 8 64 100
	WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

