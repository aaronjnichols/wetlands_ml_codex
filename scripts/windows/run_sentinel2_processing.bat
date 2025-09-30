@echo off
setlocal enabledelayedexpansion

REM ------------------------------------------------------------------
REM Edit the values below to match your project before running.
REM ------------------------------------------------------------------
set "AOI_PATH=data\small_test\test_aoi.gpkg"
set "YEARS=2022 2023 2024"
set "OUTPUT_DIR=data\small_test\s2_test"
set "SEASONS=SPR SUM FAL"
set "NAIP_PATH=data\small_test\test_naip.tif"
set "CLOUD_COVER=60"
set "MIN_CLEAR_OBS=3"
set "MASK_DILATION=0"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "LOG_LEVEL=INFO"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
)

python sentinel2_processing.py ^
    --aoi "%AOI_PATH%" ^
    --years %YEARS% ^
    --output-dir "%OUTPUT_DIR%" ^
    --seasons %SEASONS% ^
    --naip-path "%NAIP_PATH%" ^
    --cloud-cover %CLOUD_COVER% ^
    --min-clear-obs %MIN_CLEAR_OBS% ^
    --mask-dilation %MASK_DILATION% ^
    --stac-url "%STAC_URL%" ^
    --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] Sentinel-2 processing failed.
    pause
)

echo [INFO] Sentinel-2 seasonal products and stack manifest generated in %OUTPUT_DIR%
pause
