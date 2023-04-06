cmake_minimum_required(VERSION 3.13)
project(lib_XeHE)

set(LIB_SOURCES
        lib_utils.cpp
        ../include/XeHE.hpp
        include/dpcpp_utils.h
        include/lib_utils.h
        include/util/clang.h
        #include/util/common.h
        include/util/defines.h
        include/util/gcc.h
        include/util/memcache.hpp
        include/util/msvc.h
        include/util/dwt_arith.h
        include/util/rns_dwt_gpu.hpp
        include/util/rns_dwt_relin_gpu.hpp
        include/util/rns_invdwt_gpu.hpp
        include/util/rns_invdwt_relin_gpu.hpp
        include/util/rns_invdwt_rescale_gpu.hpp
        include/util/perf_gpu.hpp
        include/util/xe_ntt.h
        include/util/xe_intt_rescale.h
        include/util/xe_ntt_relin.h
        include/eval/relinearize.hpp
        include/eval/rotate.hpp        
        include/eval/xe_evaluator.hpp
        include/native/xe_uintarith_base.hpp
        include/native/xe_uintarith_core.hpp
        include/native/xe_uintarith.hpp      
        include/native/xe_polyops.hpp   
        include/native/xe_poly_kernels.hpp
        include/native/xe_poly_edge_kernels.hpp
        include/native/xe_modulus.hpp
        include/util/inline_kernels.hpp
       )

#----------------- COMMON FLAGS
set(LINK_LIBS
        "-lm;-lpthread"
        )
if (NOT MSVC)
    set(COMPILE_OPTS
#            "-MMD;-Wall;-Wextra;-Wno-unused-local-typedef;-Wno-missing-braces;-O3"
            "-MMD;-Wall;-O3"
        )
endif()

if (BUILD_WITH_IGPU)

     set(DPCPP_LIB_SOURCES
          XeHE.cpp
          dpcpp_utils.cpp
     )


    # append lib sources
    set(LIB_SOURCES
            ${LIB_SOURCES}
            ${DPCPP_LIB_SOURCES}
            )
if(WIN32)
    set(LINK_LIBS
            ${LINK_LIBS}

            "sycl;OpenCL"
    )
else()
    set(LINK_LIBS
            ${LINK_LIBS}
            "-lsycl;-lOpenCL"
            )

endif()

endif ()

set(DPCPP_APP_SOURCES
        main.cpp
        include/util/sanity_check_gpu.hpp
        )
set(PERF_APP_SOURCES
        XeHE_perf.cpp
        perf_app.hpp
        )

set(DUMP_APP_SOURCES
        XeHE_code_dump.cpp
        perf_app.hpp
        )

set(INLINE_APP_SOURCES
        XeHE_inline_asm.cpp
        )

#----------------- LIB (BOTH with DPC++ and without)
add_library(XeHE
        STATIC
        ${LIB_SOURCES}
        )

if (BUILD_WITH_IGPU)
    message(STATUS "BUILD WITH IGPU: ON")
    find_package(SYCL REQUIRED)
    message(STATUS "building with SYCL DPC++")

    #----------------- enable DPC++ components in the XeHE
    add_sycl_to_target(
            TARGET XeHE
            SOURCES ${DPCPP_LIB_SOURCES}
    )

    #----------------- SAMPLE DPC++ APP
    add_executable(dpcpp_XeHE_app
            ${DPCPP_APP_SOURCES}
            )

    add_sycl_to_target(
            TARGET dpcpp_XeHE_app
            SOURCES ${DPCPP_SOURCES}
    )

    target_link_libraries(
            dpcpp_XeHE_app
            PUBLIC ${LINK_LIBS}
            PUBLIC XeHE
    )

    target_compile_options(
            dpcpp_XeHE_app
            PRIVATE ${COMPILE_OPTS}
    )

    target_include_directories(
            dpcpp_XeHE_app
            PUBLIC "${PROJECT_SOURCE_DIR}/include"
    )

#    #----------------- PERF DPC++ APP
    add_executable(XeHE_perf
            ${PERF_APP_SOURCES}
            )
#
    add_sycl_to_target(
            TARGET XeHE_perf
            SOURCES ${PERF_APP_SOURCES}
    )
#
    target_link_libraries(
            XeHE_perf
            PUBLIC ${LINK_LIBS}
            PUBLIC XeHE
    )

    target_compile_options(
            XeHE_perf
            PRIVATE ${COMPILE_OPTS}
    )

    target_include_directories(
            XeHE_perf
            PUBLIC "${PROJECT_SOURCE_DIR}/include"
    )


#----------------- DUMP CODE DPC++ APP
add_executable(XeHE_code_dump
      ${DUMP_APP_SOURCES}
      )

add_sycl_to_target(
      TARGET XeHE_code_dump
      SOURCES ${DUMP_APP_SOURCES}
)

target_link_libraries(
      XeHE_code_dump
      PUBLIC ${LINK_LIBS}
      PUBLIC XeHE
)

target_compile_options(
      XeHE_code_dump
      PRIVATE ${COMPILE_OPTS}
)

target_include_directories(
      XeHE_code_dump
      PUBLIC "${PROJECT_SOURCE_DIR}/include"
)

#    #----------------- INLINE ASM DPC++ APP
add_executable(XeHE_inline_asm
      ${INLINE_APP_SOURCES}
      )

add_sycl_to_target(
      TARGET XeHE_inline_asm
      SOURCES ${INLINE_APP_SOURCES}
)
target_link_libraries(
      XeHE_inline_asm
      PUBLIC ${LINK_LIBS}
      PUBLIC XeHE
)

target_compile_options(
      XeHE_inline_asm
      PRIVATE ${COMPILE_OPTS}
)

target_include_directories(
      XeHE_inline_asm
      PUBLIC "${PROJECT_SOURCE_DIR}/include"
)


#    if (BUILD_WITH_SEAL)
#
#    message("find package paths: ${CMAKE_CURRENT_BINARY_DIR}/../external/SEAL/cmake ")
#    message("find package paths: ${PROJECT_SOURCE_DIR}/../external/SEAL/cmake ")
#    # This is no longer necessary. Note that find_package is usually used to find installed files,
#    # and here we just need to locate and make build tree, so find_package is an overkill.
#    #
#    #find_package (SEAL 3.6.1 EXACT REQUIRED
#    #        # Providing a path so this can be built without installing Microsoft SEAL
#    #        #PATHS ${PROJECT_SOURCE_DIR}/../external/SEAL/native/src/cmake
#    #        # It is assumed that external dir is in the parent dir
#    #        #PATHS ${CMAKE_CURRENT_BINARY_DIR}/../external/SEAL/cmake
#    #        PATHS ${PROJECT_SOURCE_DIR}/../external/SEAL/cmake
#    #        )
#
#    # append SEAL-specific includes
#    target_include_directories(
#            dpcpp_XeHE_app
#            PRIVATE "${CMAKE_SOURCE_DIR}/external/SEAL/native/src"
#    )
#
#    # append SEAL-specific link options
#    add_dependencies(dpcpp_XeHE_app seal)
#    target_link_libraries(dpcpp_XeHE_app
#            PRIVATE SEAL::seal)
##    #----------------- PERF DPC++ APP
### append SEAL-specific includes
#    target_include_directories(
#        XeHE_perf
#        PRIVATE "${CMAKE_SOURCE_DIR}/external/SEAL/native/src"
#        )
##
##        # append SEAL-specific link options
#     add_dependencies(XeHE_perf seal)
#     target_link_libraries(XeHE_perf
#        PRIVATE SEAL::seal)
##----------------- DUMP CODE DPC++ APP
## append SEAL-specific includes
#     target_include_directories(
#        XeHE_code_dump
#        PRIVATE "${CMAKE_SOURCE_DIR}/external/SEAL/native/src"
#        )
#
#        # append SEAL-specific link options
#        add_dependencies(XeHE_code_dump seal)
#        target_link_libraries(XeHE_code_dump
#                PRIVATE SEAL::seal)
#
#endif (BUILD_WITH_SEAL)

endif (BUILD_WITH_IGPU)

target_link_libraries(
        XeHE
        PUBLIC ${LINK_LIBS}
)

target_compile_options(
        XeHE
        PRIVATE ${COMPILE_OPTS}
)

target_include_directories(
        XeHE
        PUBLIC "${PROJECT_SOURCE_DIR}/include"
)