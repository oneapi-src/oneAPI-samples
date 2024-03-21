#!/bin/bash
/bin/echo "##" $(whoami) "is compiling Welcome Module-- 1 of 1 hello.cpp"
g++ src/hello.cpp -o src/hello -std=c++17
src/hello
