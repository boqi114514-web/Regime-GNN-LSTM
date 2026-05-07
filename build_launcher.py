# -*- coding: utf-8 -*-
"""
打包 launcher.py -> 启动Dashboard.exe
用法: venv\Scripts\python build_launcher.py
"""
import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / 'venv' / 'Scripts' / 'python.exe'
OUTPUT = ROOT / '启动Dashboard.exe'


def run(*args, **kwargs):
    return subprocess.run(list(args), **kwargs)


def main():
    print("=" * 50)
    print("  Regime Dashboard -- 打包启动器")
    print("=" * 50)

    if not PYTHON.exists():
        sys.exit("[错误] 找不到 venv\\Scripts\\python.exe\n"
                 "请先: python -m venv venv && venv\\Scripts\\pip install -r requirements.txt")

    # 确保 PyInstaller 已安装
    r = run(str(PYTHON), '-c', 'import PyInstaller', capture_output=True)
    if r.returncode != 0:
        print("[安装] 正在安装 PyInstaller ...")
        run(str(PYTHON), '-m', 'pip', 'install', 'pyinstaller', check=True)
        print("[安装] 完成")

    # 可选：生成图标
    icon_path = None
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGBA', (256, 256), (26, 115, 232, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([16, 16, 240, 240], fill=(13, 93, 191, 255))
        d.rectangle([80, 100, 176, 156], fill=(255, 255, 255, 220))
        d.rectangle([80, 168, 176, 184], fill=(255, 255, 255, 220))
        ico = ROOT / '_tmp_icon.ico'
        img.save(str(ico))
        icon_path = ico
        print("[图标] 已生成")
    except Exception:
        print("[图标] Pillow 未安装或出错，跳过图标")

    # 清理旧产物
    tmp = ROOT / '_pyinstaller_tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    if OUTPUT.exists():
        OUTPUT.unlink()

    # 打包
    print("\n[打包] 正在运行 PyInstaller ...")
    cmd = [
        str(PYTHON), '-m', 'PyInstaller',
        '--onefile',
        '--noconsole',
        '--name', '启动Dashboard',
        '--distpath', str(ROOT),
        '--workpath', str(tmp),
        '--specpath', str(tmp),
    ]
    if icon_path and icon_path.exists():
        cmd += ['--icon', str(icon_path)]
    cmd.append(str(ROOT / 'launcher.py'))

    r = run(*cmd)

    # 清理临时文件
    if tmp.exists():
        shutil.rmtree(tmp)
    if icon_path and icon_path.exists():
        icon_path.unlink()

    print()
    if OUTPUT.exists():
        print(f"[完成] 已生成: {OUTPUT.name}")
        print("       双击启动Dashboard.exe 即可，无需控制台。")
    else:
        print("[失败] 打包失败，请检查上方错误信息。")
        sys.exit(1)


if __name__ == '__main__':
    main()
