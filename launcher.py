# -*- coding: utf-8 -*-
"""
Thin launcher: finds Python -> runs dashboard/main.py without a console window.
Packaged with PyInstaller (--onefile --noconsole) for double-click launch.
"""
import os, sys, subprocess, ctypes
from pathlib import Path


def _msgbox(title: str, msg: str, error: bool = True):
    icon = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(0, msg, title, icon | 0x1000)


def _find_python(root: Path) -> str:
    # 1. project-local venv (venv or .venv)
    for sub in ('venv', '.venv'):
        p = root / sub / 'Scripts' / 'python.exe'
        if p.exists():
            return str(p)
    # 2. the Python that originally ran this exe (global install)
    candidate = Path(sys.executable)
    if candidate.name.lower() != 'launch_dashboard.exe':
        return str(candidate)
    # 3. system PATH
    import shutil
    found = shutil.which('python')
    if found:
        return found
    return None


def main():
    if getattr(sys, 'frozen', False):
        root = Path(sys.executable).parent
    else:
        root = Path(__file__).resolve().parent

    python = _find_python(root)
    if not python:
        _msgbox(
            'Launch failed',
            f'Could not find a Python interpreter.\n\n'
            f'Please install Python or create a venv in:\n{root}',
        )
        sys.exit(1)

    main_py = root / 'dashboard' / 'main.py'
    if not main_py.exists():
        _msgbox('Launch failed', f'dashboard/main.py not found in:\n{root}')
        sys.exit(1)

    proc = subprocess.Popen(
        [python, str(main_py)],
        cwd=str(root),
        creationflags=0x08000000,   # CREATE_NO_WINDOW
    )
    sys.exit(proc.wait())


if __name__ == '__main__':
    main()
