# -*- coding: utf-8 -*-
import os, re
from pathlib import Path
from typing import Optional
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(_THIS, "..", ".."))
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_RAW    = os.path.join(PROJECT_DIR, "data", "raw")
DATA_PROC   = os.path.join(PROJECT_DIR, "data", "processed")


# ── 工具 ──────────────────────────────────────────────────────────────────

def _read_md(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _find(text: str, pattern: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def _parse_md_table(text: str, section_pattern: str) -> list:
    """找到 section_pattern 匹配位置，解析其后的第一个 MD 表格。"""
    m = re.search(section_pattern, text)
    if not m:
        return []

    chunk = text[m.end():]
    headers: list = []
    rows: list = []
    in_table = False

    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if not stripped.startswith("|"):
            if rows:
                break
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        # 分隔行
        if all(re.fullmatch(r"[-: ]+", c) for c in cells if c):
            in_table = True
            continue

        if not headers:
            headers = cells
        elif in_table:
            if len(cells) >= len(headers):
                rows.append(dict(zip(headers, cells[:len(headers)])))

    return rows


# ── 周报 ──────────────────────────────────────────────────────────────────

def _get_last_branch() -> str:
    import json as _json
    p = os.path.join(RESULTS_DIR, "last_branch.json")
    if os.path.exists(p):
        try:
            return _json.loads(Path(p).read_text())["branch"]
        except Exception:
            pass
    return "main"


def get_report_data() -> dict:
    path = os.path.join(REPORTS_DIR, "latest.md")
    if not os.path.exists(path):
        return {"error": "报告文件不存在，请先运行周报流程"}

    text = _read_md(path)

    info = {
        "generated_at":   _find(text, r"\*\*生成时间\*\*：(.+)"),
        "iso_week":       _find(text, r"\*\*ISO 周\*\*：(.+)"),
        "data_month":     _find(text, r"\*\*数据月份\*\*：(.+)"),
        "model_version":  _find(text, r"\*\*模型版本\*\*：(.+)"),
        "regime":         _find(text, r"\*\*HMM regime\*\*：(.+)"),
        "regime_weights": _find(text, r"\*\*Regime集成权重\*\*：(.+)"),
    }

    def parse_industries(section: str) -> list:
        rows = _parse_md_table(text, rf"### {re.escape(section)}\s*\n")
        result = []
        for r in rows:
            change_raw = r.get("对比上次", "").strip()
            if "↑" in change_raw:
                direction, val = "up",   change_raw.replace("↑", "").strip()
            elif "↓" in change_raw:
                direction, val = "down", change_raw.replace("↓", "").strip()
            else:
                direction, val = "flat", "0"
            result.append({
                "rank":       r.get("排名", "").strip(),
                "code":       r.get("行业代码", "").strip(),
                "name":       r.get("行业名称", "").strip(),
                "score":      r.get("得分", "").strip(),
                "change_dir": direction,
                "change_val": val,
            })
        return result

    def parse_turnover(section: str) -> dict:
        pat = rf"### {re.escape(section)}\s*\n(.*?)(?=\n##|\n###|\Z)"
        m2 = re.search(pat, text, re.DOTALL)
        if not m2:
            return {}
        chunk = m2.group(1)
        rate   = _find(chunk, r"换手率：\*\*(.+?)\*\*")
        new_in = _find(chunk, r"新进入：(.+)")
        kicked = _find(chunk, r"被踢出：(.+)")
        return {
            "rate":   rate,
            "new_in": [x.strip() for x in new_in.split(",") if x.strip()],
            "kicked": [x.strip() for x in kicked.split(",") if x.strip()],
        }

    # ETF 表格
    etf_rows = _parse_md_table(text, r"## 🏦 ETF 执行载体\s*\n")
    etf_list = []
    for r in etf_rows:
        name_raw = r.get("ETF名称", "").strip()
        name_clean = name_raw.replace("⚠️", "").replace("⚠", "").strip()
        try:
            r2 = float(r.get("R²", "0").replace(",", "."))
        except ValueError:
            r2 = 0.0
        try:
            scale = float(r.get("规模(亿)", "0").replace(",", "."))
        except ValueError:
            scale = 0.0
        etf_list.append({
            "industry":   r.get("行业", "").strip(),
            "etf_code":   r.get("ETF代码", "").strip(),
            "etf_name":   name_clean,
            "r2":         round(r2, 3),
            "beta":       r.get("β", "").strip(),
            "scale":      round(scale, 1),
            "warn_r2":    r2 < 0.85,
            "warn_scale": scale < 2.0,
        })

    consensus = bool(re.search(r"两模型建议一致", text))

    branch_used = _get_last_branch()
    inds_regime = parse_industries("Regime 集成")
    inds_equal  = parse_industries("等权集成")
    # 主展示列表：有 Regime 集成就用，否则（refactor 分支）用等权
    industries_primary = inds_regime if inds_regime else inds_equal
    turn_regime = parse_turnover("Regime 集成")
    turn_equal  = parse_turnover("等权集成")
    turnover_primary = turn_regime if turn_regime.get("rate") else turn_equal

    return {
        **info,
        "branch_used":       branch_used,
        "industries":        industries_primary,
        "industries_regime": inds_regime,
        "industries_equal":  inds_equal,
        "etf_list":          etf_list,
        "turnover":          turnover_primary,
        "turnover_regime":   turn_regime,
        "turnover_equal":    turn_equal,
        "consensus":         consensus,
    }


# ── 个股持仓 ──────────────────────────────────────────────────────────────

def get_holdings_data() -> dict:
    path = os.path.join(REPORTS_DIR, "latest.md")
    if not os.path.exists(path):
        return {"regime": [], "equal": []}

    text = _read_md(path)

    def parse_stocks(section_re: str) -> list:
        rows = _parse_md_table(text, section_re)
        result = []
        for r in rows:
            mom = r.get("动量", "").strip()
            try:
                mom_val = float(mom.replace("%", "").replace("+", ""))
            except ValueError:
                mom_val = 0.0
            result.append({
                "industry":     r.get("行业", "").strip(),
                "code":         r.get("代码", "").strip(),
                "name":         r.get("名称", "").strip(),
                "board":        r.get("板块", "").strip(),
                "beta":         r.get("β", "").strip(),
                "momentum":     mom,
                "momentum_val": mom_val,
                "score":        r.get("综合分", "").strip(),
            })
        return result

    return {
        "regime": parse_stocks(r"### Regime 集成（\d+ 只）"),
        "equal":  parse_stocks(r"### 等权集成（\d+ 只）"),
    }


# ── 回测 ──────────────────────────────────────────────────────────────────

def _load_nav_csv(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {"dates": [], "series": {}}
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df = df.sort_values(date_col)
    dates = df[date_col].astype(str).tolist()
    series = {col: df[col].round(4).tolist() for col in df.columns[1:]}
    return {"dates": dates, "series": series}


def _load_summary_csv(filename: str) -> list:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict(orient="records")


def get_nav_data()          -> dict: return _load_nav_csv("backtest_nav.csv")
def get_backtest_summary()  -> list: return _load_summary_csv("backtest_summary.csv")
def get_etf_nav_data()      -> dict: return _load_nav_csv("etf_backtest.csv")
def get_etf_backtest_summary() -> list: return _load_summary_csv("etf_backtest_summary.csv")


# ── 历史周报轨迹 ──────────────────────────────────────────────────────────

def get_history_reports() -> list:
    results = []
    if not os.path.exists(REPORTS_DIR):
        return results
    for fname in sorted(os.listdir(REPORTS_DIR)):
        if not re.match(r"\d{4}-W\d+\.md", fname):
            continue
        try:
            text   = _read_md(os.path.join(REPORTS_DIR, fname))
            regime = _find(text, r"\*\*HMM regime\*\*：(.+)")
            week   = _find(text, r"\*\*ISO 周\*\*：(.+)") or fname.replace(".md", "")
            month  = _find(text, r"\*\*数据月份\*\*：(.+)")
            rows   = _parse_md_table(text, r"### Regime 集成\s*\n")
            top5   = [r.get("行业名称", "").strip() for r in rows[:5]]
            results.append({"week": week, "month": month, "regime": regime, "top5": top5})
        except Exception:
            pass
    return results


# ── 系统状态 ──────────────────────────────────────────────────────────────

def get_system_status() -> dict:
    targets = [
        ("raw: sw_industry", os.path.join(DATA_RAW,  "ts_sw_industry_monthly.csv")),
        ("raw: csi300",      os.path.join(DATA_RAW,  "ts_csi300_monthly.csv")),
        ("raw: macro",       os.path.join(DATA_RAW,  "ts_macro_factors.csv")),
        ("proc: prosperity", os.path.join(DATA_PROC, "prosperity_indicators_clean.pkl")),
        ("proc: pv_factors", os.path.join(DATA_PROC, "price_volume_factors.pkl")),
        ("proc: pattern",    os.path.join(DATA_PROC, "pattern_factors.pkl")),
    ]

    def _latest_month(path: str) -> Optional[str]:
        try:
            df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
            if "date" in df.columns:
                d = pd.to_datetime(df["date"], errors="coerce").max()
                return d.strftime("%Y-%m") if not pd.isna(d) else None
            if "year" in df.columns and "quarter" in df.columns:
                y = int(df["year"].max())
                q = int(df[df["year"] == df["year"].max()]["quarter"].max())
                return f"{y}-Q{q}"
        except Exception:
            pass
        return None

    data_sources = []
    for name, path in targets:
        exists = os.path.exists(path)
        latest = _latest_month(path) if exists else None
        data_sources.append({
            "name":   name,
            "exists": exists,
            "latest": latest or ("缺失" if not exists else "?"),
        })

    # 从 latest.md 提取重训时间和模型版本
    latest_report = os.path.join(REPORTS_DIR, "latest.md")
    last_retrain = model_version = infer_month = None
    if os.path.exists(latest_report):
        text = _read_md(latest_report)
        last_retrain   = _find(text, r"上次季末重训：(.+?)(?:\s*\(|\s*$)")
        model_version  = _find(text, r"\*\*模型版本\*\*：(.+)")
        infer_month    = _find(text, r"推理数据月份：(.+)")

    return {
        "data_sources":  data_sources,
        "last_retrain":  last_retrain,
        "model_version": model_version,
        "infer_month":   infer_month,
    }
