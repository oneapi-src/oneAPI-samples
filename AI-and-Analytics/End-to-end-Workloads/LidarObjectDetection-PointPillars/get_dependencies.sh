#! /bin/bash
# Build and install OpenCL ICD Loader
apt -y install ruby-full

git clone --recursive https://github.com/KhronosGroup/OpenCL-CLHPP
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
git clone https://github.com/KhronosGroup/OpenCL-Headers

cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
cmake --build ./OpenCL-Headers/build --target install

cmake -D CMAKE_PREFIX_PATH=`$PWD`/OpenCL-Headers/install -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
cmake --build ./OpenCL-ICD-Loader/build --target install

cmake -D CMAKE_PREFIX_PATH="`$PWD`/OpenCL-Headers/install;`$PWD`/OpenCL-ICD-Loader/install" -D CMAKE_INSTALL_PREFIX=./OpenCL-CLHPP/install -S ./OpenCL-CLHPP -B ./OpenCL-CLHPP/build 
cmake --build ./OpenCL-CLHPP/build --target install

export OpenCLHeadersCpp_DIR=$PWD/OpenCL-CLHPP/build/OpenCLHeadersCpp

# Install OpenVINO 2023.1.0 and Boost libraries
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list

apt update
apt -y install openvino-2023.1.0 libboost-all-dev

