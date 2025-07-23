## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(glfw3_FOUND)
    return()
endif()

if(NOT DEFINED GLFW_VERSION)
    set(GLFW_VERSION 3.3.8)
endif()

## Look for any available version
message(STATUS "Looking for glfw ${GLFW_VERSION}")
find_package(glfw3 ${GLFW_VERSION} QUIET)

if(glfw3_FOUND)
    message(STATUS "Found glfw3")
else()
    ## Download and build if not found
    set(_ARCHIVE_EXT "zip")

    if(NOT DEFINED GLFW_URL)
        set(GLFW_URL "https://github.com/glfw/glfw/releases/download/${GLFW_VERSION}/glfw-${GLFW_VERSION}.${_ARCHIVE_EXT}")
    endif()

    message(STATUS "Downloading glfw ${GLFW_URL}...")

    include(FetchContent)

    FetchContent_Declare(
        glfw
        URL "${GLFW_URL}"
        #  `patch` is not available on all systems, so use `git apply` instead.Note
        # that we initialize a Git repo in the GLFW download directory to allow the
        # Git patching approach to work.Also note that we don't want to actually
        # check out the GLFW Git repo, since we want our GLFW_HASH security checks
        # to still function correctly.
        PATCH_COMMAND git init -q . && git apply --ignore-whitespace -v -p1 < ${CMAKE_CURRENT_LIST_DIR}/glfw.patch
    )
    ## Bypass FetchContent_MakeAvailable() shortcut to disable install
    FetchContent_GetProperties(glfw)
    if(NOT glfw_POPULATED)
        FetchContent_Populate(glfw)
        message(STATUS "Adding subdirectory for glfw ${glfw_SOURCE_DIR} and ${glfw_BINARY_DIR}")

        ## the subdir will still be built since targets depend on it, but it won't be installed
        add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    message(STATUS "Adding glfw Library")

    add_library(glfw::glfw ALIAS glfw)

    unset(${_ARCHIVE_EXT})

endif()
