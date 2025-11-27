@echo off
REM Food Detection Streaming Demo Launcher
REM Double-click to run

echo ================================================
echo Food Detection Streaming Demo
echo ================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo Starting demo...
echo.

.venv11\Scripts\python.exe tests\test_streaming_demo.py

echo.
echo Demo closed.
pause
