# -*- coding: utf-8 -*-
"""
Package launcher.py -> launch_dashboard.exe
Usage: python build_launcher.py
"""
import sys, subprocess, shutil
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
PYTHON = sys.executable          # whatever python is running this script
OUTPUT = ROOT / 'launch_dashboard.exe'


def run(*args, **kwargs):
    return subprocess.run(list(args), **kwargs)


def main():
    print("=" * 50)
    print("  Regime Dashboard -- build launcher")
    print("=" * 50)
    print(f"  Python : {PYTHON}")
    print(f"  Root   : {ROOT}")

    # ensure PyInstaller is installed
    r = run(PYTHON, '-c', 'import PyInstaller', capture_output=True)
    if r.returncode != 0:
        print("[INFO] Installing PyInstaller ...")
        run(PYTHON, '-m', 'pip', 'install', 'pyinstaller', check=True)

    # optional icon via Pillow
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
        print("[INFO] Icon generated")
    except Exception:
        print("[INFO] Pillow not available, skipping icon")

    # clean previous build
    tmp = ROOT / '_pyinstaller_tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    if OUTPUT.exists():
        OUTPUT.unlink()

    # run PyInstaller
    print("\n[BUILD] Running PyInstaller ...")
    cmd = [
        PYTHON, '-m', 'PyInstaller',
        '--onefile',
        '--noconsole',
        '--name', 'launch_dashboard',
        '--distpath', str(ROOT),
        '--workpath', str(tmp),
        '--specpath', str(tmp),
    ]
    if icon_path and icon_path.exists():
        cmd += ['--icon', str(icon_path)]
    cmd.append(str(ROOT / 'launcher.py'))

    run(*cmd)

    # cleanup
    if tmp.exists():
        shutil.rmtree(tmp)
    if icon_path and icon_path.exists():
        icon_path.unlink()

    print()
    if OUTPUT.exists():
        print(f"[DONE] {OUTPUT.name} created")
        print("       Double-click to launch Dashboard without a console window.")
    else:
        print("[FAIL] Build failed. Check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
