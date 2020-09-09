cmake_minimum_required(VERSION 3.1)
project(basic-pub-sub CXX)

option(BUILD_DEPS "Builds aws common runtime dependencies as part of build to control your dependency chain." ON)

if (DEFINED CMAKE_PREFIX_PATH)
    file(TO_CMAKE_PATH "${CMAKE_PREFIX_PATH}" CMAKE_PREFIX_PATH)
endif()

if (DEFINED CMAKE_INSTALL_PREFIX)
    file(TO_CMAKE_PATH "${CMAKE_INSTALL_PREFIX}" CMAKE_INSTALL_PREFIX)
endif()

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif()

if (UNIX AND NOT APPLE)
    include(GNUInstallDirs)
elseif(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (${CMAKE_INSTALL_LIBDIR} STREQUAL "lib64")
    set(FIND_LIBRARY_USE_LIB64_PATHS true)
endif()

# This is required in order to append /lib/cmake to each element in CMAKE_PREFIX_PATH
set(AWS_MODULE_DIR "/${CMAKE_INSTALL_LIBDIR}/cmake")
string(REPLACE ";" "${AWS_MODULE_DIR};" AWS_MODULE_PATH "${CMAKE_PREFIX_PATH}${AWS_MODULE_DIR}")
# Append that generated list to the module search path
list(APPEND CMAKE_MODULE_PATH ${AWS_MODULE_PATH})

if (NOT DEFINED CMAKE_BUILD_TYPE)
    if (NOT WIN32)
        set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/sdk-cpp-workspace/aws-iot-device-sdk-cpp-v2/aws-common-runtime/aws-crt-cpp/aws-common-runtime/aws-c-common/cmake")

include(AwsFindPackage)
set(IN_SOURCE_BUILD ON)
set(BUILD_TESTING_PREV ${BUILD_TESTING})
set(BUILD_TESTING OFF)

find_path(CRT_CPP_LIB aws-crt-cpp PATHS "$ENV{HOME}/sdk-cpp-workspace/aws-iot-device-sdk-cpp-v2/aws-common-runtime")
set(CRT_PATH ${CRT_CPP_LIB}/aws-crt-cpp)
add_subdirectory(${CRT_PATH} build EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${BUILD_TESTING_PREV})


list(APPEND CMAKE_MODULE_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake")

file(GLOB PUB_SUB_SRC
       "cpp/*.cpp"
)

set(PUB_SUB_PROJECT_NAME basic-pub-sub)
add_executable(${PUB_SUB_PROJECT_NAME} ${PUB_SUB_SRC})
set_target_properties(${PUB_SUB_PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

set(CMAKE_C_FLAGS_DEBUGOPT "")

#set warnings
if (MSVC)
    target_compile_options(${PUB_SUB_PROJECT_NAME} PRIVATE /W4 /WX /wd4068)
else ()
    target_compile_options(${PUB_SUB_PROJECT_NAME} PRIVATE -Wall -Wno-long-long -pedantic -Werror)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "" OR CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PUB_SUB_PROJECT_NAME} PRIVATE "-DDEBUG_BUILD")
endif ()

target_include_directories(${PUB_SUB_PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

aws_use_package(aws-crt-cpp)
target_link_libraries(${PUB_SUB_PROJECT_NAME} ${DEP_AWS_LIBS})

