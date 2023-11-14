@echo off

if ""%1"" == ""clean"" goto clean

echo Building Tachyon.common project 
Rem batch build to build Tachyon.common.lib
Rem Output is created in "tachyon_release" directory
if exist "tachyon_release" goto build_start
mkdir tachyon_release
goto build_start

:clean
echo cleaning lib build files
if exist "tachyon_release" goto clean2
exit /b

:clean2
del /F /Q tachyon_release\*.*
exit /b

:build_start

set CC=icl
set LK=xilink

set CFLAGS=/c /O2 /Ot /FD /EHsc /MD /Gy /arch:SSE2 /fp:fast /Zi /W2 /TP /Qdiag-disable:10210
set DefFlags=/D "DEFAULT_MODELFILE=balls.dat" /D "EMULATE_PTHREADS" /D "WIN32" /D "_WINDOWS" /D "NDEBUG" /D "_CRT_SECURE_NO_DEPRECATE" /D "_MBCS"
set OutDir=tachyon_release\\
set OutFlags=/Fo"%OutDir%" /Fd"%OutDir%vc80.pdb"  
set srcDir=.\src\
set commonDir=.\src\common\gui\

echo rc.exe /fo"%OutDir%gui.res" .\msvs2017\gui.rc
rc.exe /fo"%OutDir%gui.res" .\msvs2017\gui.rc

echo icl %CFLAGS% %DefFlags% %OutFlags% %srcDir%api.cpp %srcDir%vol.cpp %srcDir%tachyon_video.cpp %srcDir%vector.cpp %srcDir%util.cpp %srcDir%ui.cpp %srcDir%triangle.cpp %srcDir%trace_rest.cpp %srcDir%tgafile.cpp %srcDir%texture.cpp %srcDir%sphere.cpp %srcDir%shade.cpp %srcDir%ring.cpp %srcDir%render.cpp %srcDir%quadric.cpp %srcDir%pthread.cpp %srcDir%ppm.cpp %srcDir%plane.cpp %srcDir%parse.cpp %srcDir%objbound.cpp %srcDir%light.cpp %srcDir%jpeg.cpp %srcDir%intersect.cpp %srcDir%imap.cpp %srcDir%imageio.cpp %srcDir%grid.cpp %srcDir%global.cpp %srcDir%getargs.cpp %srcDir%extvol.cpp %srcDir%cylinder.cpp %srcDir%coordsys.cpp %srcDir%camera.cpp %srcDir%box.cpp %srcDir%bndbox.cpp %srcDir%apitrigeom.cpp %srcDir%apigeom.cpp %commonDir%gdivideo.cpp
icl %CFLAGS% %DefFlags% %OutFlags% %srcDir%*.cpp %commonDir%gdivideo.cpp

echo xilib.exe /OUT:"%OutDir%tachyon.common.lib" /LTCG .\%OutDir%api.obj .\%OutDir%apigeom.obj .\%OutDir%apitrigeom.obj .\%OutDir%bndbox.obj .\%OutDir%box.obj .\%OutDir%camera.obj .\%OutDir%coordsys.obj .\%OutDir%cylinder.obj .\%OutDir%extvol.obj .\%OutDir%getargs.obj .\%OutDir%global.obj .\%OutDir%grid.obj .\%OutDir%imageio.obj .\%OutDir%imap.obj .\%OutDir%intersect.obj .\%OutDir%jpeg.obj .\%OutDir%light.obj .\%OutDir%objbound.obj .\%OutDir%parse.obj .\%OutDir%plane.obj .\%OutDir%ppm.obj .\%OutDir%pthread.obj .\%OutDir%quadric.obj .\%OutDir%render.obj .\%OutDir%ring.obj .\%OutDir%shade.obj .\%OutDir%sphere.obj .\%OutDir%texture.obj .\%OutDir%tgafile.obj .\%OutDir%trace_rest.obj .\%OutDir%triangle.obj .\%OutDir%ui.obj .\%OutDir%util.obj .\%OutDir%vector.obj .\%OutDir%tachyon_video.obj .\%OutDir%vol.obj .\%OutDir%gui.res .\%OutDir%gdivideo.obj
xilib.exe /OUT:"%OutDir%tachyon.common.lib" /LTCG .\%OutDir%*.obj .\%OutDir%gui.res
