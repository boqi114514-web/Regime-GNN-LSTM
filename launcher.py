# -*- coding: utf-8 -*-
"""
启动 Dashboard 的薄壳程序。
定位 venv/Scripts/python.exe → 运行 dashboard/main.py，无控制台窗口。
PyInstaller 打包后放在项目根目录即可双击使用。
"""
import os, sys, subprocess, ctypes
from pathlib import Path


def _msgbox(title: str, msg: str, error: bool = True):
    icon = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(0, msg, title, icon | 0x1000)


def _find_python(root: Path) -> Path | None:
    for sub in ('venv', '.venv'):
        p = root / sub / 'Scripts' / 'python.exe'
        if p.exists():
            return p
    return None


def main():
    # 打包后 sys.executable 是 .exe 本身；脚本模式则取文件目录
    if getattr(sys, 'frozen', False):
        root = Path(sys.executable).parent
    else:
        root = Path(__file__).resolve().parent

    python = _find_python(root)
    if python is None:
        _msgbox(
            '启动失败',
            f'未找到虚拟环境（venv 或 .venv）。\n\n'
            f'请先在项目目录下执行：\n'
            f'  python -m venv venv\n'
            f'  venv\\Scripts\\pip install -r requirements.txt\n\n'
            f'项目路径：{root}',
        )
        sys.exit(1)

    main_py = root / 'dashboard' / 'main.py'
    if not main_py.exists():
        _msgbox('启动失败', f'找不到 dashboard/main.py\n\n项目路径：{root}')
        sys.exit(1)

    # CREATE_NO_WINDOW：不弹黑色控制台
    proc = subprocess.Popen(
        [str(python), str(main_py)],
        cwd=str(root),
        creationflags=0x08000000,
    )
    sys.exit(proc.wait())


if __name__ == '__main__':
    main()
