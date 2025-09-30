@echo off
setlocal enabledelayedexpansion

set "STACK_MANIFEST="data\small_test\s2\stack_manifest.json""
set "TRAIN_RASTER="
set "LABELS=data\small_test\wetlands.gpkg"
set "TILES_DIR=data\small_test\tiles"
set "MODELS_DIR=data\small_test\models"
set "TILE_SIZE=128"
set "STRIDE=64"
set "BUFFER=0"
set "BATCH_SIZE=4"
set "EPOCHS=10"
set "LEARNING_RATE=0.0001"
set "VAL_SPLIT=0.1"
set "NUM_CHANNELS="
set "PRETRAINED=true"
set "LOG_LEVEL=INFO"

if "%LABELS%"=="" (
    echo [ERROR] LABELS must point to your wetlands training data.
    pause
    exit /b 1
)

if "%STACK_MANIFEST%"=="" if "%TRAIN_RASTER%"=="" (
    echo [ERROR] Provide STACK_MANIFEST or TRAIN_RASTER before running.
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
if not "%TRAIN_RASTER%"=="" set "RASTER_ARG=--train-raster \"%TRAIN_RASTER%\""

set "NUM_CHANNELS_ARG="
if not "%NUM_CHANNELS%"=="" set "NUM_CHANNELS_ARG=--num-channels %NUM_CHANNELS%"

if /I "%PRETRAINED%"=="false" (
    set "PRETRAINED_FLAG=--no-pretrained"
) else (
    set "PRETRAINED_FLAG=--pretrained"
)

python train.py ^
    %STACK_ARG% %RASTER_ARG% ^
    --labels "%LABELS%" ^
    --tiles-dir "%TILES_DIR%" ^
    --models-dir "%MODELS_DIR%" ^
    --tile-size %TILE_SIZE% ^
    --stride %STRIDE% ^
    --buffer %BUFFER% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --learning-rate %LEARNING_RATE% ^
    --val-split %VAL_SPLIT% ^
    --log-level %LOG_LEVEL% ^
    %NUM_CHANNELS_ARG% ^
    %PRETRAINED_FLAG%

if errorlevel 1 (
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo [INFO] Training complete. Models saved to %MODELS_DIR%
pause
exit /b 0
