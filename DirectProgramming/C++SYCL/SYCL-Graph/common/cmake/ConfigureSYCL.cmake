############################################################################
## Copyright © 2025 Codeplay Software
##
## SPDX-License-Identifier: MIT
############################################################################

# ------------------------------------------------
# Detect available backends
# ------------------------------------------------
execute_process(
    COMMAND bash -c "! sycl-ls | grep -q cuda"
    RESULT_VARIABLE CUDA_BACKEND_AVAILABLE)
execute_process(
    COMMAND bash -c "! sycl-ls | grep -q hip"
    RESULT_VARIABLE HIP_BACKEND_AVAILABLE)
execute_process(
    COMMAND bash -c "! sycl-ls | grep -q 'opencl\\|level_zero'"
    RESULT_VARIABLE SPIR_BACKEND_AVAILABLE)

set(ENABLE_CUDA ${CUDA_BACKEND_AVAILABLE} CACHE BOOL "Build with CUDA target")
set(ENABLE_HIP ${HIP_BACKEND_AVAILABLE} CACHE BOOL "Build with HIP target")
set(ENABLE_SPIR ${SPIR_BACKEND_AVAILABLE} CACHE BOOL "Build with spir64 target")
set(SYCL_TARGETS "")

# ------------------------------------------------
# Configure CUDA target
# ------------------------------------------------
if(${ENABLE_CUDA})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "nvptx64-nvidia-cuda")
    set(DEFAULT_CUDA_COMPUTE_CAPABILITY "50")
    set(CUDA_COMPUTE_CAPABILITY "" CACHE BOOL
        "CUDA architecture (compute capability), e.g. sm_80. Default value is auto-configured using nvidia-smi.")
    # Auto-configure if not specified by user
    if ("${CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
        execute_process(
            COMMAND bash -c "which nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.'"
            OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    # Warn if not specified and failed to auto-configure
    if ("${CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
        message(WARNING "Failed to autoconfigure CUDA_COMPUTE_CAPABILITY using nvidia-smi. Will default to sm_${DEFAULT_CUDA_COMPUTE_CAPABILITY}")
        set(CUDA_COMPUTE_CAPABILITY ${DEFAULT_CUDA_COMPUTE_CAPABILITY} CACHE STRING "CUDA Compute Capability")
    else()
        message(STATUS "Enabled SYCL target CUDA with Compute Capability sm_${CUDA_COMPUTE_CAPABILITY}")
    endif()
endif()

# ------------------------------------------------
# Configure HIP target
# ------------------------------------------------
if(${ENABLE_HIP})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "amdgcn-amd-amdhsa")
    set(DEFAULT_HIP_GFX_ARCH "gfx906")
    set(HIP_GFX_ARCH "" CACHE BOOL
        "HIP architecture tag, e.g. gfx90a. Default value is auto-configured using rocminfo.")
    # Auto-configure if not specified by user
    if ("${CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
        execute_process(
            COMMAND bash -c "which rocminfo >/dev/null && rocminfo | grep -o 'gfx[0-9]*' | head -n 1"
            OUTPUT_VARIABLE HIP_GFX_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    # Warn if not specified and failed to auto-configure
    if ("${HIP_GFX_ARCH}" STREQUAL "")
        message(WARNING "Failed to autoconfigure HIP_GFX_ARCH using rocminfo. Will default to ${DEFAULT_HIP_GFX_ARCH}")
        set(HIP_GFX_ARCH ${DEFAULT_HIP_GFX_ARCH} CACHE STRING "HIP gfx arch")
    else()
        message(STATUS "Enabled SYCL target HIP with gfx arch ${HIP_GFX_ARCH}")
    endif()
endif()

# ------------------------------------------------
# Configure spir64 target
# ------------------------------------------------
if(${ENABLE_SPIR})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "spir64")
    message(STATUS "Enabled SYCL target spir64")
endif()

# ------------------------------------------------
# Configure the complete SYCL flags
# ------------------------------------------------
set(SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})
if(${ENABLE_CUDA})
    list(APPEND SYCL_FLAGS -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${CUDA_COMPUTE_CAPABILITY})
endif()
if(${ENABLE_HIP})
    list(APPEND SYCL_FLAGS -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_GFX_ARCH})
endif()
