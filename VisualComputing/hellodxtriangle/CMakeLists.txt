cmake_minimum_required(VERSION 3.19)
project(studyTutorial
	DESCRIPTION "Study hello world program for drawing ospTutorial with DirectX"
  LANGUAGES CXX)

#Download and install directxtk from vcpkg (approx 600MB)
#If you see:
#  status: [4;"A requested feature, protocol or option was not found built-in in this libcurl due to a build-time decision."] OR
#  status: [5;"Couldn't resolve proxy name"]
#  Make sure you set HTTPS_PROXY to the proxy server for your organization as needed: set HTTPS_PROXY=http://your.proxy.com:###
set(VCPKG_URL "https://github.com/microsoft/vcpkg/archive/refs/tags/2021.05.12.zip")
set(VCPKG_MD5 "a81b4205c3ab1c53ba879cfa3386b1bf")
set(VCPKG_DIR "${CMAKE_BINARY_DIR}/vcpkg-2021.05.12")
if(NOT EXISTS "${VCPKG_DIR}/bootstrap-vcpkg.bat")

  message("Downloading ${VCPKG_URL} to vcpkg.zip... if you see a hang here try configuiring your proxy (HTTPS_PROXY HTTP_PROXY)")

  file(DOWNLOAD ${VCPKG_URL} ${CMAKE_BINARY_DIR}/vcpkg.zip EXPECTED_MD5 ${VCPKG_MD5} STATUS VCPKG_DOWNLOAD_STATUS SHOW_PROGRESS)
  message("vcpkg download status: ${VCPKG_DOWNLOAD_STATUS}")

  file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/vcpkg.zip DESTINATION ${CMAKE_BINARY_DIR})
  message("vcpkg extracted")
else()


  message("vcpkg already downloaded")
endif()

set(VCPKG_INSTALL_OUTPUT "")
set(VCPKG_RESULT_OUTPUT "")
if(NOT EXISTS "${VCPKG_DIR}/installed/x64-windows/share/directxtk/directxtk-config.cmake")
  message("vcpkg bootstrapping and installing directxtk:x64-windows directxmath:x64-windows")
  set(BOOTSTRAP_BAT ${VCPKG_DIR}\\bootstrap-vcpkg.bat)
  execute_process(
    COMMAND ${BOOTSTRAP_BAT} 
    WORKING_DIRECTORY ${VCPKG_DIR}
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE
  )

  set(VCPKG_EXE ${VCPKG_DIR}\\vcpkg.exe)
  set(VCPKG_ARGS install directxtk:x64-windows directxmath:x64-windows)
  execute_process(
    COMMAND ${VCPKG_EXE} ${VCPKG_ARGS}
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE
  )

else()

  message("directxtoolkit from vcpkg already installed")
endif()

message("vcpkg bootstrapped")

add_executable(hellodxtriangle
  hellodxtriangle.cpp
)

#Configure the target/project
target_compile_definitions(hellodxtriangle PRIVATE "$<$<CONFIG:DEBUG>:_DEBUG>" "$<$<CONFIG:RELEASE>:NDEBUG>")

set(directxtk_DIR ${VCPKG_DIR}/installed/x64-windows/share/directxtk)
set(directxmath_DIR ${VCPKG_DIR}/installed/x64-windows/share/directxmath)
find_package(directxtk CONFIG REQUIRED)
find_package(directxmath CONFIG REQUIRED)
target_link_libraries(hellodxtriangle PRIVATE
  d3d11.lib
  Microsoft::DirectXTK)
target_compile_options(hellodxtriangle PRIVATE /Zc:__cplusplus)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT hellodxtriangle)
