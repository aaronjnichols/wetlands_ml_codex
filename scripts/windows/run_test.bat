@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Edit the values below before running inference.
REM Leave STACK_MANIFEST empty if you prefer to use TEST_RASTER.
REM ------------------------------------------------------------------
set "STACK_MANIFEST=data\small_test\s2_test\stack_manifest.json"
set "TEST_RASTER="
set "MODEL_PATH=data\small_test\models\best_model.pth"
set "OUTPUT_DIR=data\small_test\predictions"
set "MASK_PATH=data\small_test\predictions\mini_mask.tif"
set "VECTOR_PATH=data\small_test\predictions\mini_predictions.gpkg"
set "WINDOW_SIZE=256"
set "OVERLAP=128"
set "BATCH_SIZE=4"
set "CONFIDENCE=0.25"
set "NUM_CHANNELS="
set "MIN_AREA=100"
set "SIMPLIFY=0.25"
set "LOG_LEVEL=INFO"

if "%MODEL_PATH%"=="" (
    echo [ERROR] MODEL_PATH must point to a trained .pth file.
    pause
    exit /b 1
)

if "%STACK_MANIFEST%"=="" if "%TEST_RASTER%"=="" (
    echo [ERROR] Provide STACK_MANIFEST or TEST_RASTER before running.
    pause
    exit /b 1
)

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

set "STACK_ARG="
if not "%STACK_MANIFEST%"=="" set "STACK_ARG=--stack-manifest \"%STACK_MANIFEST%\""

set "RASTER_ARG="
if not "%TEST_RASTER%"=="" set "RASTER_ARG=--test-raster \"%TEST_RASTER%\""

set "MASK_ARG="
if not "%MASK_PATH%"=="" set "MASK_ARG=--masks \"%MASK_PATH%\""

set "VECTOR_ARG="
if not "%VECTOR_PATH%"=="" set "VECTOR_ARG=--vectors \"%VECTOR_PATH%\""

set "NUM_CHANNELS_ARG="
if not "%NUM_CHANNELS%"=="" set "NUM_CHANNELS_ARG=--num-channels %NUM_CHANNELS%"

python test.py ^
    %STACK_ARG% %RASTER_ARG% ^
    --model-path "%MODEL_PATH%" ^
    --output-dir "%OUTPUT_DIR%" ^
    %MASK_ARG% %VECTOR_ARG% ^
    --window-size %WINDOW_SIZE% ^
    --overlap %OVERLAP% ^
    --confidence-threshold %CONFIDENCE% ^
    --batch-size %BATCH_SIZE% ^
    %NUM_CHANNELS_ARG% ^
    --min-area %MIN_AREA% ^
    --simplify-tolerance %SIMPLIFY% ^
    --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] Inference failed.
    pause
    exit /b 1
)

echo [INFO] Inference complete. Outputs in %OUTPUT_DIR%
pause
exit /b 0
