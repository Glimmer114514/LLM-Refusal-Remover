@echo off
set "VENV=%~dp0.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] 虚拟环境不存在: %VENV%
    echo 请先运行: python -m venv "%VENV%"
    pause
    exit /b 1
)

set "PYTHON=%VENV%\Scripts\python.exe"
set "PATH=%VENV%\Scripts;%PATH%"

if "%~1"=="" (
    echo === Model Brain Surgery 虚拟环境 ===
    echo Python: %PYTHON%
    %PYTHON% --version
    echo.
    echo 可用命令:
    echo   run.bat surgery          运行完整手术
    echo   run.bat surgery-test     轻量smoke test
    echo   run.bat chat             聊天测试
    echo   run.bat python script.py 运行自定义脚本
    echo.
    cmd /k
    exit /b
)

if "%~1"=="surgery" (
    %PYTHON% "%~dp0brain_surgery.py"
    exit /b %ERRORLEVEL%
)

if "%~1"=="surgery-test" (
    %PYTHON% "%~dp0brain_surgery.py" --layers 2,3 --ablation-scale 0.1 --skip-save
    exit /b %ERRORLEVEL%
)

if "%~1"=="chat" (
    %PYTHON% "%~dp0chat_qwen.py"
    exit /b %ERRORLEVEL%
)

if "%~1"=="python" (
    shift
    %PYTHON% %*
    exit /b %ERRORLEVEL%
)

%PYTHON% %*
exit /b %ERRORLEVEL%
