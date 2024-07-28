@echo off

if ""%1"" == ""clean"" goto clean

echo Building build_serial project 
rem for building win32 release config only
rem build "build_serial" program
rem output is created in "tachyon_release" directory
if exist "tachyon_release" goto build_start
mkdir tachyon_release
goto build_start

:clean
echo cleaning serial build files
if exist "tachyon_release" goto clean2
exit /b

:clean2
del /F /Q tachyon_release\*.*
exit /b

:build_start

set CFLAGS=/c /O2 /Oi /Ot /Qipo /EHsc /MD /GS /arch:SSE2 /fp:fast /W2 /Zi /I src
set DefFlags=/D "WIN32" /D "_WINDOWS" /D "NDEBUG" /D "_MBCS" 
set OutDir=.\tachyon_release\\
set OutFlags=/Fo"%OutDir%" 
set srcFName=tachyon.serial
set outFName=tachyon.serial
set srcDir=.\src\tachyon.serial\

echo icl %CFLAGS% %DefFlags% %OutFlags% %srcDir%%srcFName%.cpp
icl %CFLAGS% %DefFlags% %OutFlags% %srcDir%%srcFName%.cpp

echo xilink.exe /OUT:"%OutDir%%outFName%.exe" /INCREMENTAL:NO /MANIFEST /MANIFESTFILE:"%OutDir%%outFName%.exe.intermediate.manifest" /TLBID:1 /DEBUG /PDB:"%OutDir%%outFName%.pdb" /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF  /IMPLIB:"%OutDir%%outFName%.lib" /FIXED:NO %OutDir%tachyon.common.lib %OutDir%%srcFName%.obj
xilink.exe /OUT:"%OutDir%%outFName%.exe" /INCREMENTAL:NO /MANIFEST /MANIFESTFILE:"%OutDir%%outFName%.exe.intermediate.manifest" /TLBID:1 /DEBUG /PDB:"%OutDir%%outFName%.pdb" /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF  /IMPLIB:"%OutDir%%outFName%.lib" /FIXED:NO %OutDir%tachyon.common.lib %OutDir%%srcFName%.obj

echo mt.exe /outputresource:"%OutDir%%outFName%.exe;#1" /manifest %OutDir%%outFName%.exe.intermediate.manifest
mt.exe /outputresource:"%OutDir%%outFName%.exe;#1" /manifest %OutDir%%outFName%.exe.intermediate.manifest
