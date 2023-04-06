cmake_minimum_required (VERSION 3.13)
project(HEAAN_cmake LANGUAGES CXX)
# 	g++ -I/usr/local/include -O3 -c -std=c++11 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (HEAAN_LIB_SOURCES
    ${PROJECT_SOURCE_DIR}/src/BootContext.cpp
    ${PROJECT_SOURCE_DIR}/src/Ciphertext.cpp
    ${PROJECT_SOURCE_DIR}/src/EvaluatorUtils.cpp
    ${PROJECT_SOURCE_DIR}/src/Key.cpp
    ${PROJECT_SOURCE_DIR}/src/Plaintext.cpp
    ${PROJECT_SOURCE_DIR}/src/Ring.cpp
    ${PROJECT_SOURCE_DIR}/src/RingMultiplier.cpp
    ${PROJECT_SOURCE_DIR}/src/Scheme.cpp
    ${PROJECT_SOURCE_DIR}/src/SchemeAlgo.cpp
    ${PROJECT_SOURCE_DIR}/src/SecretKey.cpp
    ${PROJECT_SOURCE_DIR}/src/SerializationUtils.cpp
    ${PROJECT_SOURCE_DIR}/src/StringUtils.cpp
    ${PROJECT_SOURCE_DIR}/src/TestScheme.cpp
    ${PROJECT_SOURCE_DIR}/src/TimeUtils.cpp 
    )

add_library (
    obj_HEAAN
    OBJECT
    ${HEAAN_LIB_SOURCES}
    )

include_directories (/usr/local/include)

set_property (TARGET obj_HEAAN PROPERTY POSITION_INDEPENDENT_CODE ON)

set (HEAAN_COMPILE_OPTS
    "-MMD;-Wno-missing-braces;-O3"
    )

target_compile_options (obj_HEAAN 
    PRIVATE ${HEAAN_COMPILE_OPTS})

add_library (
    HEAAN_lib
    STATIC
    $<TARGET_OBJECTS:obj_HEAAN>
    )

target_link_libraries (
    HEAAN_lib
    PUBLIC Threads::Threads
    )

add_subdirectory (run)
