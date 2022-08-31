include(SetLibraryPath)

add_executable(matrix matrix.cpp multiply.cpp)
target_include_directories(matrix PRIVATE ${DEVUTILITIES_INCLUDE_DIR})
