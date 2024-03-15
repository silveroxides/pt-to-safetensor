@echo off

set PYTHON=python
set "VENV_DIR=%~dp0%venv"

%PYTHON% -mpip --help > nul 2> nul
if %ERRORLEVEL% == 0 goto :start_venv
echo Couldn't install pip
goto :endofscript

:start_venv
dir "%VENV_DIR%\Scripts\Python.exe" > nul 2> nul
if %ERRORLEVEL% == 0 goto :activate_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" > nul 2> nul
if %ERRORLEVEL% == 0 goto :setup

echo Unable to create venv in directory "%VENV_DIR%"
goto :endofscript

:setup
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%
%PYTHON% -m pip install -r %~dp0requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% == 0 goto :launch
echo.
echo Setup unsuccessful. Exiting.
pause
exit /b

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%

:launch
%PYTHON% %~dp0bin-pt_to_safetensors.py %*
pause
exit /b

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
