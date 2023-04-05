@echo off
call "C:\Program Files (x86)\Intel\oneapi\setvars.bat"

REM set C_COMPILER=dpcpp
REM set CXX_COMPILER=dpcpp

set "BUILD_COMMAND=cmake -G \"Visual Studio 17 2022\" -A x64 -T "Intel(R) oneAPI DPC++ Compiler 2023" -D CMAKE_INSTALL_PREFIX=.. .."
echo %BUILD_COMMAND:\=% > build-command.txt
echo "CXX_COMPILER:" >> build-command.txt
%CXX_COMPILER% --version >> build-command.txt

rmdir /S /Q build
mkdir build
pushd build
%BUILD_COMMAND:\=%
cmake --build . --config Release
cmake --install .
popd
copy build-command.txt bin\build-command.txt
