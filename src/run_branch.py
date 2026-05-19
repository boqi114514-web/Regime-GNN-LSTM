# -*- coding: utf-8 -*-
"""
切换到指定分支，运行完整流水线（run_all.py），生成周报（monitor.run()），
跑完后恢复调用时所在分支。由 dashboard/api/runner.py 作为子进程调用。

用法: python run_branch.py <branch_name>
"""
import sys, os, subprocess, json, runpy, importlib.util as ilu

ALLOWED = {
    "main",
    "fix/macro-neutral-fill",
    "refactor/equal-weight-ensemble",
}

def main():
    branch = sys.argv[1] if len(sys.argv) > 1 else "main"
    if branch not in ALLOWED:
        print(f"[ERROR] 未知分支: {branch}  (允许: {', '.join(sorted(ALLOWED))})")
        sys.exit(1)

    src_dir  = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 记录调用时所在分支，跑完恢复（不固定 main）
    orig_proc = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=proj_dir, capture_output=True, text=True,
    )
    orig_branch = (orig_proc.stdout.strip() or 'main')
    print(f"[分支] 调用时所在分支: {orig_branch}")

    # 暂存本地改动，防止 checkout 被拒
    subprocess.run(['git', 'stash'], cwd=proj_dir, capture_output=True)

    try:
        r = subprocess.run(
            ['git', 'checkout', branch],
            cwd=proj_dir, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[ERROR] git checkout 失败:\n{r.stderr.strip()}")
            sys.exit(1)
        print(f"[分支] 已切换 → {branch}")

        # 清除已缓存的项目模块，确保加载最新代码
        stale = [m for m in list(sys.modules)
                 if any(m.startswith(p) for p in
                        ('config', 's0_', 's1_', 's2_', 's3_', 'live', 'data_pipeline'))]
        for m in stale:
            del sys.modules[m]

        # ── 运行完整流水线 ────────────────────────────────────────────────
        runpy.run_path(os.path.join(src_dir, 'run_all.py'), run_name='__main__')

        # ── 生成周报 ──────────────────────────────────────────────────────
        print('\n[周报] 生成周报...')
        spec = ilu.spec_from_file_location(
            'monitor', os.path.join(src_dir, 'live', 'monitor.py'))
        mon = ilu.module_from_spec(spec)
        spec.loader.exec_module(mon)
        mon.run()
        print('[OK] 周报已更新')

        # ── 记录本次所用分支（供前端展示）────────────────────────────────
        results_dir = os.path.join(proj_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'last_branch.json'), 'w') as f:
            json.dump({'branch': branch}, f)

    finally:
        subprocess.run(['git', 'checkout', orig_branch], cwd=proj_dir, capture_output=True)
        subprocess.run(['git', 'stash', 'pop'],          cwd=proj_dir, capture_output=True)
        print(f'[OK] 已恢复 {orig_branch} 分支')


if __name__ == '__main__':
    main()
