
set(CMAKE_CXX_COMPILER icpx)

cmake_minimum_required(VERSION 3.4)
project(Tachyon)

option(DEBUG "Add debug information" ON)


set(OMP_EXE "tachyon.openmp")
set(TBB_EXE "tachyon.tbb")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(COMMON_FLAGS "")
set(COMMON_LIBS "-lm")


# Detect UI ----------------------------------------------------------------------
find_library(X libX11.so)

if(NOT X STREQUAL "X-NOTFOUND") # graphics output detected
    set(COMMON_FLAGS ${COMMON_FLAGS} -I${X} -DX_FULLSYNC)
    set(COMMON_LIBS "${COMMON_LIBS} -lpthread -L/usr/X11R6/lib64 -lX11")
    set(UI "x")

    find_library(LIBXEXT libXext.so)

    if(NOT LIBXEXT STREQUAL "LIBXEXT-NOTFOUND") # libXext detected
        set(COMMON_LIBS "${COMMON_LIBS} -lXext")
        set(COMMON_FLAGS "${COMMON_FLAGS} -DX_NOSHMPIX")
    else() # no libXext
        set(COMMON_FLAGS "${COMMON_FLAGS} -DX_NOSHMEM")
    endif() #libxext

else() # no graphics output
    set(UI "con")
    message(WARNING " Note: no graphics output capability detected, building for console output.")
endif() # graphics
# --------------------------------------------------------------------------------------------

if(DEBUG) 
    set(CoMMON_FLAGS -O0 -g -D_DEBUG ${COMMON_FLAGS})
    set(COMMON_LIBS -O0 -g ${COMMON_LIBS})
else()
    set(COMMON_FLAGS -O2 -g ${COMMON_FLAGS})
    set(COMMON_LIBS -O2 -g ${COMMON_LIBS})
endif()



include_directories(include)
file(GLOB COMMON_SRC CONFIGURE_DEPENDS "src/*.cpp")


set(UI_SRC src/common/gui/${UI}video.cpp)
set(SERIAL_SRC ${CMAKE_SOURCE_DIR}/src/build_with_serial/build_with_serial.cpp)
set(OMP_SRC ${CMAKE_SOURCE_DIR}/src/build_with_openmp/build_with_openmp.cpp)
set(TBB_SRC ${CMAKE_SOURCE_DIR}/src/build_with_tbb/build_with_tbb.cpp)
set(OMP_SLN_SRC ${CMAKE_SOURCE_DIR}/src/build_with_openmp/build_with_openmp_optimized.cpp)
set(TBB_SLN_SRC ${CMAKE_SOURCE_DIR}/src/build_with_tbb/build_with_tbb_optimized.cpp)

# generate targets that generate executables
add_executable(tachyon.serial ${SERIAL_SRC} ${UI_SRC} ${COMMON_SRC})
add_executable(${OMP_EXE} ${OMP_SRC} ${UI_SRC} ${COMMON_SRC})
add_executable(${TBB_EXE} ${TBB_SRC} ${UI_SRC} ${COMMON_SRC})
add_executable(${OMP_EXE}_optimized ${OMP_SLN_SRC} ${UI_SRC} ${COMMON_SRC})
add_executable(${TBB_EXE}_optimized ${TBB_SLN_SRC} ${UI_SRC} ${COMMON_SRC})


# Add libraries and and compiler options -------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(COMMON_LIBS -lrt ${COMMON_LIBS})
endif()


target_compile_options(tachyon.serial PUBLIC ${COMMON_FLAGS})
target_link_libraries(tachyon.serial ${COMMON_LIBS})

foreach(EXE IN ITEMS ${OMP_EXE} ${OMP_EXE}_optimized)
    target_compile_options(${EXE} PUBLIC -qopenmp ${COMMON_FLAGS})
    target_link_libraries(${EXE} -qopenmp ${COMMON_LIBS})
endforeach(EXE IN ITEMS)


foreach(EXE IN ITEMS ${TBB_EXE} ${TBB_EXE}_optimized)

    if(DEBUG)
        target_compile_options(${EXE} PUBLIC -tbb -DTBB_USE_DEBUG -qtbb ${COMMON_FLAGS})
        target_link_libraries(${EXE} -ltbb_debug ${COMMON_LIBS})
    else()
        target_compile_options(${EXE} PUBLIC -tbb ${COMMON_FLAGS})
        target_link_libraries(${EXE} -tbb ${COMMON_LIBS})
    endif()
    
endforeach(EXE IN ITEMS ${TBB_EXE} ${TBB_EXE}_optimized)



# Custom targets to run exec(s) -------------------------------------------------------------

set(DATA_FILE ../dat/balls.dat)

add_custom_target(run
COMMAND echo "== Running all versions"
COMMAND echo "== Serial"
COMMAND tachyon.serial ${DATA_FILE}
COMMAND echo "== OpenMP"
COMMAND tachyon.openmp ${DATA_FILE} 
COMMAND tachyon.openmp_optimized ${DATA_FILE}
COMMAND echo "== TBB"
COMMAND tachyon.tbb ${DATA_FILE}
COMMAND tachyon.tbb_optimized ${DATA_FILE}
DEPENDS tachyon.serial tachyon.openmp tachyon.openmp_optimized tachyon.tbb tachyon.tbb_optimized
WORKING_DIRECTORY ${CMAKE_PROJECT_DIR})

foreach(VERSION IN ITEMS serial openmp tbb openmp_optimized tbb_optimized)
    add_custom_target(${VERSION}
    COMMAND echo "== Running ${VERSION} version"
    COMMAND tachyon.${VERSION} ${DATA_FILE}
    DEPENDS tachyon.${VERSION}
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
endforeach(VERSION IN ITEMS)

# Custom targets to run solution exec(s)

add_custom_target(solution
COMMAND tachyon.serial ${DATA_FILE}
COMMAND tachyon.openmp_optimized ${DATA_FILE} 
COMMAND tachyon.tbb_optimized ${DATA_FILE}
DEPENDS tachyon.serial tachyon.openmp_optimized tachyon.tbb_optimized
WORKING_DIRECTORY ${CMAKE_PROJECT_DIR})
