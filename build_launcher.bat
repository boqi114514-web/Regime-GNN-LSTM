@echo off
chcp 65001 > nul
setlocal

set ROOT=%~dp0
set PYTHON=%ROOT%venv\Scripts\python.exe
set DIST=%ROOT%

echo ================================================
echo  Regime Dashboard -- 打包启动器
echo ================================================
echo.

:: 检查 venv
if not exist "%PYTHON%" (
    echo [错误] 找不到 venv\Scripts\python.exe
    echo 请先创建虚拟环境：python -m venv venv
    pause & exit /b 1
)

:: 安装 / 确认 PyInstaller
"%PYTHON%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [安装] 正在安装 PyInstaller...
    "%PYTHON%" -m pip install pyinstaller --quiet
)

:: 生成图标（需要 Pillow；没有则跳过）
set ICON_ARG=
"%PYTHON%" -c "from PIL import Image, ImageDraw; import os; img=Image.new('RGBA',(256,256),(26,115,232,255)); d=ImageDraw.Draw(img); d.ellipse([20,20,236,236],fill=(255,255,255,40)); img.save(r'%ROOT%_launcher_icon.ico')" 2>nul
if exist "%ROOT%_launcher_icon.ico" (
    set ICON_ARG=--icon=%ROOT%_launcher_icon.ico
    echo [图标] 已生成图标
) else (
    echo [图标] Pillow 未安装，跳过图标（使用默认）
)

:: 清理旧产物
if exist "%ROOT%build_tmp" rd /s /q "%ROOT%build_tmp"
if exist "%ROOT%启动Dashboard.exe" del /f /q "%ROOT%启动Dashboard.exe"

:: PyInstaller 打包
echo.
echo [打包] 正在打包...
"%PYTHON%" -m PyInstaller ^
    --onefile ^
    --noconsole ^
    --name "启动Dashboard" ^
    --distpath "%DIST%" ^
    --workpath "%ROOT%build_tmp" ^
    --specpath "%ROOT%build_tmp" ^
    %ICON_ARG% ^
    "%ROOT%launcher.py"

:: 清理临时文件
if exist "%ROOT%build_tmp"           rd  /s /q "%ROOT%build_tmp"
if exist "%ROOT%_launcher_icon.ico"  del /f /q "%ROOT%_launcher_icon.ico"

echo.
if exist "%ROOT%启动Dashboard.exe" (
    echo [完成] 已生成：启动Dashboard.exe
    echo        双击即可启动 Dashboard，无需控制台。
) else (
    echo [失败] 打包失败，请检查上方错误信息。
    pause
)
echo.
endlocal
pause
