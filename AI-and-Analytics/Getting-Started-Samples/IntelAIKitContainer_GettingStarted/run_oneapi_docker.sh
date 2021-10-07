if [ -z "$1" ] ; then
    echo "Usage:  $0    <image name>    [optional command]"
    echo "Missing Docker image id.  exiting"
    exit -1
fi

image_id="$1"
name="aikit_container"

## remove any previously running containers
docker rm -f "$name"

# mount the current directory at /work
this="${BASH_SOURCE-$0}"
mydir=$(cd -P -- "$(dirname -- "$this")" && pwd -P)

export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} -e SOCKS_PROXY=${SOCKS_PROXY}"

docker run --privileged $DOCKER_RUN_ENVS -dit  --name "$name" \
    -p 8888:8888 \
    -p 6006:6006 \
    -v"/home:/home" \
    --net host \
    -p 6543:6543 \
    -p 12345:12345 \
    "$image_id"
docker exec -it "$name" /bin/bash -c "pip install matplotlib"
docker exec -it "$name" /bin/bash -c "apt-get update -yq;apt-get install -yq vim numactl"
docker exec -it "$name" /bin/bash
