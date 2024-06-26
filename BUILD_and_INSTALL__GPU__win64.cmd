@echo off

ECHO.

SET "MYHOME=C:\Users\Installed_packages\GTalign"
SET /p MYHOME=Enter GTalign install path (Default: %MYHOME%): 

ECHO. & ECHO Install path: %MYHOME% & ECHO.

IF NOT EXIST build_win64 (mkdir build_win64 || EXIT /b)
CD build_win64 || EXIT /b

cmake -DGPUINUSE=1 -DCMAKE_INSTALL_PREFIX=%MYHOME% ^
    -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75" ^
    ../src/  ||  (CD .. & EXIT /b)

cmake --build . --config Release  ||  (CD .. & EXIT /b)
cmake --install .  --config Release  ||  (CD .. & EXIT /b)

CD ..
EXIT /b
