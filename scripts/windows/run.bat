@echo off
setlocal

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo WARNING: venv not found. Ensure dependencies are installed on PATH.
)

echo Running Sentinel-2 + NAIP 25-band pipeline...
python sentinel2_processing.py ^
    --aoi "data\train_aoi.gpkg" ^
    --years 2023 ^
    --output-dir "data\s2_test3" ^
    --naip-path "data\naip\m_2708230_nw_17_030_20230207_20230511.tif"

endlocal
pause
