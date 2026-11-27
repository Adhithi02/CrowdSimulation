@echo off
REM ===========================================================================
REM  Build the CUDA crowd plugin and the native benchmark harness.
REM
REM  Requirements: CUDA Toolkit (tested 13.5), Visual Studio C++ tools
REM  (tested VS 18 / MSVC v145), an NVIDIA GPU (tested RTX 2050, sm_86).
REM
REM  Usage (from this folder, any terminal):
REM     build.bat              build into .\build\
REM     build.bat bench        build, then run the Phase 3 benchmarks
REM     build.bat install      build, then copy the DLL into the Unity project
REM                            (close the Unity Editor first: it locks the DLL)
REM
REM  Override the GPU architecture with:  set CUDA_ARCH=sm_75  (before running)
REM ===========================================================================
setlocal
cd /d "%~dp0"
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_86

REM -- Locate MSVC (nvcc needs cl.exe on PATH) --------------------------------
REM (no parenthesised blocks here: the ")" in "%ProgramFiles(x86)%" would end them)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
where cl >nul 2>nul && goto :have_cl
if not exist "%VSWHERE%" goto :no_cl
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -products * -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" goto :no_cl
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
where cl >nul 2>nul && goto :have_cl
:no_cl
echo [ERROR] MSVC cl.exe not found. Install Visual Studio "Desktop development with C++".
exit /b 1
:have_cl
where nvcc >nul 2>nul || (echo [ERROR] nvcc not found. Install the CUDA Toolkit. & exit /b 1)

if not exist build mkdir build
set NVFLAGS=-O3 -arch=%CUDA_ARCH% -std=c++17 -allow-unsupported-compiler -Xcompiler "/MD /EHsc" -I.

echo [1/2] CrowdSimulationCUDA.dll  (%CUDA_ARCH%)
nvcc %NVFLAGS% -shared CrowdPlugin.cu CrowdKernels.cu -o build\CrowdSimulationCUDA.dll || exit /b 1

echo [2/2] run_phase3.exe
nvcc %NVFLAGS% run_phase3.cpp build\CrowdSimulationCUDA.lib -o build\run_phase3.exe || exit /b 1

echo Build OK: %CD%\build
if /i "%1"=="bench" (
    pushd build
    .\run_phase3.exe || (popd & exit /b 1)
    popd
    echo Results: build\Phase3_Trials.csv, build\Phase3_Statistics.csv
)
if /i "%1"=="install" (
    copy /y build\CrowdSimulationCUDA.dll "..\CSIM\Assets\Plugins\x86_64\CrowdSimulationCUDA.dll" || (echo [ERROR] Copy failed - is Unity open? & exit /b 1)
    echo Installed into CSIM\Assets\Plugins\x86_64\
)
endlocal
