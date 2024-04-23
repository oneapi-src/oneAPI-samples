#!/bin/bash

export COMMON_VOICE_PATH="/data/commonVoice"
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} -e SOCKS_PROXY=${SOCKS_PROXY} -e COMMON_VOICE_PATH=${COMMON_VOICE_PATH}"
docker run --privileged ${DOCKER_RUN_ENVS} -it --rm --network host \
    -v"/home:/home" \
    -v"/tmp:/tmp" \
    -v "${PWD}/Inference":/Inference \
    -v "${PWD}/Training":/Training \
    -v "${COMMON_VOICE_PATH}":/data \
    --shm-size 32G \
    intel/oneapi-aikit
    