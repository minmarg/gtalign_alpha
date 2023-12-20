@echo off

ECHO.

SET "MYHOME=C:\Users\Installed_packages\GTalign_mp_win64"
SET /p MYHOME=Enter GTalign install path (Default: %MYHOME%): 

ECHO. & ECHO Install path: %MYHOME% & ECHO.

IF NOT EXIST build_mp_win64 (mkdir build_mp_win64 || EXIT /b)
CD build_mp_win64 || EXIT /b

cmake ^
    -DGPUINUSE=0 -DFASTMATH=1 -DCMAKE_INSTALL_PREFIX=%MYHOME% -DCMAKE_VERBOSE_MAKEFILE=ON ^
    ../src/  ||  (CD .. & EXIT /b)

cmake --build . --config Release  ||  (CD .. & EXIT /b)
cmake --install .  --config Release  ||  (CD .. & EXIT /b)

CD ..
EXIT /b
