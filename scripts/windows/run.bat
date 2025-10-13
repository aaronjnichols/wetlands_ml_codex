@echo off
setlocal

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM Activate virtual environment if it exists
if not exist "venv\Scripts\activate.bat" (
    echo WARNING: venv not found. Ensure dependencies are installed on PATH.
) else (
    call "venv\Scripts\activate.bat"
)

set "AOI_PATH=data\train_aoi.gpkg"
set "YEARS=2023"
set "OUTPUT_DIR=data\s2_test3"
set "NAIP_PATH=data\naip\m_2708230_nw_17_030_20230207_20230511.tif"
set "AUTO_DOWNLOAD_NAIP=false"
set "AUTO_DOWNLOAD_NAIP_YEAR=2023"
set "AUTO_DOWNLOAD_NAIP_MAX_ITEMS="
set "AUTO_DOWNLOAD_NAIP_OVERWRITE=false"
set "AUTO_DOWNLOAD_NAIP_PREVIEW=false"
set "AUTO_DOWNLOAD_WETLANDS=false"
set "WETLANDS_OUTPUT=data\wetlands\wetlands_auto.gpkg"
set "WETLANDS_OVERWRITE=false"

set "NAIP_ARG="
if not "%NAIP_PATH%"=="" set "NAIP_ARG=--naip-path \"%NAIP_PATH%\""

set "AUTO_NAIP_FLAG="
set "AUTO_NAIP_YEAR_ARG="
set "AUTO_NAIP_MAX_ITEMS_ARG="
set "AUTO_NAIP_OVERWRITE_FLAG="
set "AUTO_NAIP_PREVIEW_FLAG="
if /I "%AUTO_DOWNLOAD_NAIP%"=="true" (
    set "AUTO_NAIP_FLAG=--auto-download-naip"
    if not "%AUTO_DOWNLOAD_NAIP_YEAR%"=="" set "AUTO_NAIP_YEAR_ARG=--auto-download-naip-year %AUTO_DOWNLOAD_NAIP_YEAR%"
    if not "%AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"=="" set "AUTO_NAIP_MAX_ITEMS_ARG=--auto-download-naip-max-items %AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"
    if /I "%AUTO_DOWNLOAD_NAIP_OVERWRITE%"=="true" set "AUTO_NAIP_OVERWRITE_FLAG=--auto-download-naip-overwrite"
    if /I "%AUTO_DOWNLOAD_NAIP_PREVIEW%"=="true" set "AUTO_NAIP_PREVIEW_FLAG=--auto-download-naip-preview"
)

set "AUTO_WETLANDS_FLAG="
set "WETLANDS_OUTPUT_ARG="
set "WETLANDS_OVERWRITE_FLAG="
if /I "%AUTO_DOWNLOAD_WETLANDS%"=="true" (
    set "AUTO_WETLANDS_FLAG=--auto-download-wetlands"
    if not "%WETLANDS_OUTPUT%"=="" set "WETLANDS_OUTPUT_ARG=--wetlands-output-path \"%WETLANDS_OUTPUT%\""
    if /I "%WETLANDS_OVERWRITE%"=="true" set "WETLANDS_OVERWRITE_FLAG=--wetlands-overwrite"
)

echo Running Sentinel-2 + NAIP 25-band pipeline...
python "sentinel2_processing.py" ^
    --aoi "%AOI_PATH%" ^
    --years %YEARS% ^
    --output-dir "%OUTPUT_DIR%" ^
    %NAIP_ARG% ^
    %AUTO_NAIP_FLAG% ^
    %AUTO_NAIP_YEAR_ARG% ^
    %AUTO_NAIP_MAX_ITEMS_ARG% ^
    %AUTO_NAIP_OVERWRITE_FLAG% ^
    %AUTO_NAIP_PREVIEW_FLAG% ^
    %AUTO_WETLANDS_FLAG% ^
    %WETLANDS_OUTPUT_ARG% ^
    %WETLANDS_OVERWRITE_FLAG%

endlocal
pause
