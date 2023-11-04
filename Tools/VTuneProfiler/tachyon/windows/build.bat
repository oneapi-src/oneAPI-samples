@echo off 

call scripts\buildlib.bat clean
call scripts\buildserial.bat clean
call scripts\buildtbb.bat clean
call scripts\buildopenmp.bat clean
if ""%1"" ==  ""clean"" exit /b


if ""%1"" == ""optimized"" goto build_optimized
if ""%1"" == ""buildserial"" goto buildserial
if ""%1"" == ""buildtbb"" goto buildtbb
if ""%1"" == ""buildopenmp"" goto buildopenmp

if ""%1"" == ""openmp_optimized"" goto openmp_optimized
if ""%1"" == ""tbb_optimized"" goto tbb_optimized

:build_start
call scripts\buildlib.bat
call scripts\buildserial.bat
call scripts\buildtbb.bat
call scripts\buildopenmp.bat
exit /b

:buildserial
call scripts\buildlib.bat
call scripts\buildserial.bat
exit /b

:buildtbb
call scripts\buildlib.bat
call scripts\buildtbb.bat
exit /b

:buildopenmp
call scripts\buildlib.bat
call scripts\buildopenmp.bat
exit /b

:openmp_optimized
call scripts\buildlib.bat
call scripts\buildopenmp.bat optimized
exit /b

:tbb_optimized
call scripts\buildlib.bat
call scripts\buildtbb.bat optimized
exit /b

:build_optimized
echo Building Solutions
call scripts\buildlib.bat
call scripts\buildserial.bat
call scripts\buildtbb.bat optimized
call scripts\buildopenmp.bat optimized
