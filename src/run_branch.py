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

# 跨分支共享的基础设施：选股层(s4) + 报告(monitor) 在所有分支应保持一致，
# 分支差异只应体现在 regime/集成模型(s0-s3)。切到目标分支后用 main 版覆盖，
# 保证每个分支都用同一套（含主板过滤、估值因子、主板持仓章节）的选股+报告逻辑。
SHARED_FROM_MAIN = [
    "src/s4_beta_selection.py",
    "src/live/monitor.py",
]


def _overlay_shared(proj_dir):
    """用 main 分支版本覆盖共享文件（s4 + monitor）"""
    for rel in SHARED_FROM_MAIN:
        r = subprocess.run(['git', 'show', f'main:{rel}'],
                           cwd=proj_dir, capture_output=True)
        if r.returncode == 0:
            with open(os.path.join(proj_dir, rel), 'wb') as f:
                f.write(r.stdout)
            print(f"[共享] 已用 main 版覆盖 {rel}")
        else:
            print(f"[共享] 跳过 {rel}（git show 失败）")


def _restore_shared(proj_dir):
    """丢弃共享文件覆盖，恢复当前分支的提交版本（避免后续 checkout 被拒）"""
    subprocess.run(['git', 'checkout', '--', *SHARED_FROM_MAIN],
                   cwd=proj_dir, capture_output=True)

def main():
    branch = sys.argv[1] if len(sys.argv) > 1 else "main"
    if branch not in ALLOWED:
        print(f"[ERROR] 未知分支: {branch}  (允许: {', '.join(sorted(ALLOWED))})")
        sys.exit(1)

    src_dir  = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 跑完始终恢复到 main：dashboard 只能在 main 分支上正常工作
    # （分支切换架构、持仓页主板 tab 等都在 main），固定恢复 main 避免卡在功能分支
    orig_branch = 'main'
    print(f"[分支] 跑完将恢复到: {orig_branch}")

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

        # 用 main 的共享 s4 + monitor 覆盖（非 main 分支才需要）
        if branch != 'main':
            _overlay_shared(proj_dir)

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
        if branch != 'main':
            _restore_shared(proj_dir)   # 先丢弃共享覆盖，否则 checkout 会被拒
        subprocess.run(['git', 'checkout', orig_branch], cwd=proj_dir, capture_output=True)
        subprocess.run(['git', 'stash', 'pop'],          cwd=proj_dir, capture_output=True)
        print(f'[OK] 已恢复 {orig_branch} 分支')


if __name__ == '__main__':
    main()
