## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(glm_FOUND)
    return()
endif()

if(NOT DEFINED GLM_VERSION)
    set(GLM_VERSION 0.9.9.8)
endif()

## Look for any available version
message(STATUS "Looking for glm ${GLM_VERSION}")
find_package(glm ${GLM_VERSION} QUIET)

if(glm_FOUND)
    message(STATUS "Found glm")
else()
    ## Download and build if not found
    set(_ARCHIVE_EXT "zip")

    if(NOT DEFINED GLM_URL)
        set(GLM_URL "https://github.com/g-truc/glm/releases/download/${GLM_VERSION}/glm-${GLM_VERSION}.${_ARCHIVE_EXT}")
    endif()

    message(STATUS "Downloading glm ${GLM_URL}...")

    include(FetchContent)

    FetchContent_Declare(
        glm
        URL "${GLM_URL}"
        #  `patch` is not available on all systems, so use `git apply` instead.Note
        # that we initialize a Git repo in the GLM download directory to allow the
        # Git patching approach to work.Also note that we don't want to actually
        # check out the GLM Git repo, since we want our GLM_HASH security checks
        # to still function correctly.
        PATCH_COMMAND git init -q . && git apply --ignore-whitespace -v -p1 < ${CMAKE_CURRENT_LIST_DIR}/glm.patch
    )
    ## Bypass FetchContent_MakeAvailable() shortcut to disable install
    FetchContent_GetProperties(glm)
    if(NOT glm_POPULATED)
        FetchContent_Populate(glm)
        message(STATUS "Adding subdirectory for glm ${glm_SOURCE_DIR} and ${glm_BINARY_DIR}")

        ## the subdir will still be built since targets depend on it, but it won't be installed
        add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    message(STATUS "Adding glm Library")

    add_library(glm::glm ALIAS glm)

    unset(${_ARCHIVE_EXT})

endif()
