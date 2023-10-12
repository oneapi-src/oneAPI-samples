#!/bin/bash

git clone https://github.com/libsdl-org/SDL.git
cd SDL

git checkout release-2.0.20
mkdir build
mkdir install
cd build
cmake ../ -D CMAKE_INSTALL_PREFIX=../install
make && make install

