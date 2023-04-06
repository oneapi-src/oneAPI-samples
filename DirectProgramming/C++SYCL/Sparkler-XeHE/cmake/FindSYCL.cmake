cmake_minimum_required(VERSION 3.13)
include(FindPackageHandleStandardArgs)
# NOTE That 3.12+ is required for func
#    add_sycl_to_target to work for OBJECT

find_path(
    SYCL_INCLUDE_DIR
    NAMES
        "CL/sycl.hpp"
    PATHS
        ENV CMPLR_ROOT
        ENV ONEAPI_ROOT
    PATH_SUFFIXES
        "linux/lib/clang/9.0.0/include"
        "windows/lib/clang/9.0.0/include"
        "compiler/latest/linux/lib/clang/9.0.0/include/"
        "compiler/latest/windows/lib/clang/9.0.0/include"
        "linux/lib/clang/10.0.0/include"
        "windows/lib/clang/10.0.0/include"
        "linux/lib/clang/11.0.0/include"
        "windows/lib/clang/11.0.0/include"
        "compiler/latest/linux/lib/clang/10.0.0/include/"
        "compiler/latest/linux/lib/clang/11.0.0/include/"
        "compiler/latest/windows/lib/clang/10.0.0/include"
        "compiler/latest/windows/lib/clang/11.0.0/include"
	"compiler/latest/linux/include/sycl/"
	"compiler/latest/windows/include/sycl/"
    NO_DEFAULT_PATH
)

if(NOT WIN32)
    find_package_handle_standard_args(
        SYCL
        DEFAULT_MSG
        SYCL_INCLUDE_DIR
    )
else()
    find_path(
        SYCL_IMPORT_LIBRARY_DIR
        NAMES
            "sycl.lib"
        PATHS
            ENV CMPLR_ROOT
            ENV ONEAPI_ROOT
        PATH_SUFFIXES
            "windows/lib"
            "compiler/latest/windows/lib"
        NO_DEFAULT_PATH
    )

    find_package_handle_standard_args(
        SYCL
        DEFAULT_MSG
        SYCL_INCLUDE_DIR
        SYCL_IMPORT_LIBRARY_DIR
    )
endif()

set(SYCL_INCLUDE_DIRS "${SYCL_INCLUDE_DIR}")

function(add_sycl_to_target)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(SYCL "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")

    #set(CompileFlags "-g0  -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-sycldevice")
    set(CompileFlags "-g0  -fsycl-unnamed-lambda")
    #set(CompileFlags "-O1 -fsycl-unnamed-lambda  -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \\\"-device Gen12HP\\\"")
    #set(CompileFlags "-fsycl-unnamed-lambda  -fsycl-targets=spir64_gen-unknown-unknown-sycldevice")
    #set(CompileFlags "-g0 -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-unknown-sycldevice")
    #set(CompileFlags "-g0 -fsycl-unnamed-lambda -fsycl-device-code-split=off")
    #set(CompileFlags "-g0 -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -fsycl-device-code-split=off")

    set(TargetLinkFlags "")

    if(WIN32)
        set(CompileFlags "${CompileFlags}")
        set(TargetLinkFlags "sycl;OpenCL")
    else()
        set(CompileFlags "${CompileFlags} -Wno-unused-parameter")
        set(TargetLinkFlags "-fsycl")
    endif()

    foreach(source_file ${SYCL_SOURCES})
        target_sources("${SYCL_TARGET}" PRIVATE "${source_file}")
        # -g0 flag is used to avoid linking issues with sycl in debug mode
        set_source_files_properties("${source_file}" PROPERTIES COMPILE_FLAGS "${CompileFlags}")
    endforeach(source_file)

    target_include_directories("${SYCL_TARGET}" SYSTEM PUBLIC "${SYCL_INCLUDE_DIRS}")
    target_link_libraries("${SYCL_TARGET}" PUBLIC "${TargetLinkFlags}")

    if(WIN32)
        target_link_directories("${SYCL_TARGET}" PUBLIC "${SYCL_IMPORT_LIBRARY_DIR}")
    else()
        target_compile_options("${SYCL_TARGET}" PUBLIC "${TargetLinkFlags}")
    endif()
endfunction(add_sycl_to_target)
