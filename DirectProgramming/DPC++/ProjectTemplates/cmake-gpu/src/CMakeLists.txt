IF(CMAKE_BUILD_TYPE MATCHES "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
ELSE()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
ENDIF()

add_executable(cmake-gpu main.cpp)

add_custom_target(build
  ALL
  DEPENDS cmake-gpu
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

add_custom_target(run
  COMMAND ../cmake-gpu
  DEPENDS build
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)