cmake_minimum_required(VERSION 2.8.11)
project(azure-iothub-telemetry)

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set (CMAKE_C_FLAGS "--std=c99 ${CMAKE_C_FLAGS}")
        if (NOT IN_OPENWRT)
            set (CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
        endif()
    endif()
else()
    set (CMAKE_C_STANDARD 99)
    set (CMAKE_CXX_STANDARD 11)
endif()

set(iothub_c_files
    cpp/iothub_ll_telemetry_sample.c
)

if(${use_sample_trusted_cert})
    add_definitions(-DSET_TRUSTED_CERT_IN_SAMPLES)
    include_directories(certs)
    set(iothub_c_files certs/certs.c)
endif()

set(AZUREIOT_INC_FOLDER "." "/usr/include/azureiot" "/usr/include/azureiot/inc")

include_directories(${AZUREIOT_INC_FOLDER})

add_executable(azure-iothub-telemetry ${iothub_c_files})
target_link_libraries(azure-iothub-telemetry
    iothub_client_mqtt_transport
    iothub_client_amqp_transport
    iothub_client_http_transport
    iothub_client
    umqtt
    prov_auth_client
    hsm_security_client
    uhttp
    aziotsharedutil
    parson
    uuid
    pthread
    curl
    ssl
    crypto
    m
)

add_subdirectory(cpp)

add_custom_target (run ./azure-iothub-telemetry)

