# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(TEST_SOURCE
    ${CMAKE_CURRENT_LIST_DIR}/ciphertext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ckks.cpp
    ${CMAKE_CURRENT_LIST_DIR}/context.cpp
    ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.cpp
    ${CMAKE_CURRENT_LIST_DIR}/encryptor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/evaluator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/galoiskeys.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynarray.cpp
    ${CMAKE_CURRENT_LIST_DIR}/keygenerator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/memorymanager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/modulus.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plaintext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/publickey.cpp
    ${CMAKE_CURRENT_LIST_DIR}/randomgen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/randomtostd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/relinkeys.cpp
    ${CMAKE_CURRENT_LIST_DIR}/secretkey.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/testrunner.cpp
)

target_sources(sealtest
    PRIVATE
    ${TEST_SOURCE}
)

if(SEAL_USE_INTEL_XEHE)
    set(TEST_SOURCE
            ${TEST_SOURCE}
            ${CMAKE_CURRENT_LIST_DIR}/evaluator_gpu.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/evaluator_gpu.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/seal_ckks_kernel_executor.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/seal_ckks_context.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/test_seal_ckks_kernel_executor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/seal_omp_utils.cpp

	    )
    add_sycl_to_target(
            TARGET sealtest
            SOURCES ${TEST_SOURCE}
    )
endif(SEAL_USE_INTEL_XEHE)

add_subdirectory(util)
