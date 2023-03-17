@echo off
call "C:\Program Files (x86)\Intel\oneapi\setvars.bat"

set CXX_COMPILER=icx-cl

set "BUILD_COMMAND=cmake -G\"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=%CXX_COMPILER% -DCMAKE_INSTALL_PREFIX=.. .."
echo %BUILD_COMMAND:\=% > build-command.txt
echo "CXX_COMPILER:" >> build-command.txt
%CXX_COMPILER% --version >> build-command.txt

rmdir /S /Q build
mkdir build
pushd build
%BUILD_COMMAND:\=%
cmake --build .
cmake --install .
popd
copy build-command.txt bin\build-command.txt
