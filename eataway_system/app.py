"""
Eataway 预测系统 — Web 界面 (生产版本)

本地运行:  python app.py
云端部署:  gunicorn --chdir eataway_system app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1

可选密码保护: 设置环境变量 APP_PASSWORD=你的密码
  → 上传数据和触发训练需要密码；查看数据不需要

日期选择功能:
  - 选择过去的日期 → 显示 1year.csv 里的真实发货数据（FAKTISK）
  - 选择未来/当前周 → 显示模型预测结果（PROGNOS）
"""

import subprocess
import threading
import re
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 最大上传 100MB

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
OUTPUT_DIR     = BASE_DIR / "output_v7"
FEATURE_SCRIPT = BASE_DIR / "feature.py"
TRAIN_SCRIPT   = BASE_DIR / "eataway_train_v7.py"
DATA_FILE      = BASE_DIR / "1year.csv"
DATA_ZIP       = BASE_DIR / "1year.csv.zip"   # GitHub 友好备用（6.8 MB）

def _ensure_data_file():
    """若 1year.csv 不存在但 1year.csv.zip 存在，则自动解压。"""
    if not DATA_FILE.exists() and DATA_ZIP.exists():
        import zipfile
        try:
            with zipfile.ZipFile(DATA_ZIP, "r") as zf:
                zf.extract("1year.csv", BASE_DIR)
            print(f"✓ 已自动解压 {DATA_ZIP.name} → 1year.csv")
        except Exception as e:
            print(f"✗ 解压失败: {e}")

# ── 可选密码 ──────────────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD", "eataway")

def _password_ok() -> bool:
    if not APP_PASSWORD:
        return True
    pw = (request.headers.get("X-App-Password")
          or request.args.get("pw")
          or (request.get_json(silent=True) or {}).get("pw")
          or request.form.get("pw"))
    return pw == APP_PASSWORD

# ── 流水线状态 ────────────────────────────────────────────────
pipeline_status = {"running": False, "log": [], "done": False, "error": False}


# ── 日期工具 ──────────────────────────────────────────────────
MONTHS_SV   = ["jan","feb","mar","apr","maj","jun",
                "jul","aug","sep","okt","nov","dec"]
DAY_OFFSETS = {"Sunday":-1,"Monday":0,"Tuesday":1,"Wednesday":2,
               "Thursday":3,"Friday":4,"Saturday":5}
DAY_SHORT   = {"Sunday":"Sön","Monday":"Mån","Tuesday":"Tis","Wednesday":"Ons",
               "Thursday":"Tor","Friday":"Fre","Saturday":"Lör"}
DAY_ORDER   = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,
               "Thursday":4,"Friday":5,"Saturday":6}

def week_monday(year_week: str):
    m = re.match(r"(\d{4})-W(\d{2})", str(year_week))
    if not m:
        return None
    y, w = int(m.group(1)), int(m.group(2))
    return datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")

def fmt_d(d: datetime) -> str:
    return f"{d.day} {MONTHS_SV[d.month-1]}"

def week_to_range(yw: str) -> str:
    mon = week_monday(yw)
    if not mon:
        return yw
    sun = mon - timedelta(days=1)
    thu = mon + timedelta(days=3)
    return f"{fmt_d(sun)} – {fmt_d(thu)} {mon.year}"

def day_label(yw: str, day_name: str) -> str:
    mon = week_monday(yw)
    if not mon:
        return day_name
    d = mon + timedelta(days=DAY_OFFSETS.get(day_name, 0))
    return f"{DAY_SHORT.get(day_name, day_name)} {d.day}/{d.month}"

def week_step(yw: str, delta: int) -> str:
    """Advance a year_week string by delta weeks."""
    m = re.match(r"(\d{4})-W(\d{2})", yw)
    if not m:
        return yw
    mon = week_monday(yw)
    if not mon:
        return yw
    target = mon + timedelta(weeks=delta)
    iso = target.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── 流水线 ────────────────────────────────────────────────────
def run_pipeline_thread():
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": False}
    python = sys.executable
    steps = [
        ("特征工程", [python, str(FEATURE_SCRIPT)]),
        ("模型训练", [python, str(TRAIN_SCRIPT)]),
    ]
    for step_name, cmd in steps:
        pipeline_status["log"].append(f"\n▶ {step_name}...")
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(BASE_DIR),
                env=env,
            )
            for line in proc.stdout:
                pipeline_status["log"].append(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                pipeline_status["log"].append(f"✗ {step_name} 失败 (code={proc.returncode})")
                pipeline_status["error"] = True
                break
            else:
                pipeline_status["log"].append(f"✓ {step_name} 完成")
        except Exception as e:
            pipeline_status["log"].append(f"✗ 错误: {e}")
            pipeline_status["error"] = True
            break
    pipeline_status["running"] = False
    pipeline_status["done"]    = True


# ── 模型预测数据（来自 output_v7/ CSV） ───────────────────────
def load_csv(path):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None

def get_predicted_week() -> str:
    """返回模型输出的周（driver_view_v7.csv 中的最大周）。"""
    for fname in ["driver_view_v7.csv", "evaluation_v7.csv"]:
        df = load_csv(OUTPUT_DIR / fname)
        if df is not None and "year_week" in df.columns:
            return str(df["year_week"].max())
    return ""

def get_driver_predicted(display_yw: str = "") -> dict:
    """从模型输出 CSV 获取司机视图。display_yw 用于覆盖日期标签。"""
    df = load_csv(OUTPUT_DIR / "driver_view_v7.csv")
    if df is None:
        return {}
    yw = display_yw or (str(df["year_week"].max()) if "year_week" in df.columns else "")
    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    df = df.sort_values(["Route", "_ord", "Store"])
    result = {}
    for route, rdf in df.groupby("Route"):
        result[route] = {}
        for store, sdf in rdf.groupby("Store"):
            result[route][store] = []
            for _, row in sdf.sort_values("_ord").iterrows():
                result[route][store].append({
                    "day":      row["Day"],
                    "day_label": day_label(yw, row["Day"]),
                    "qty":      int(row["Total_Qty"]),
                    "range":    row["Range"],
                    "products": str(row.get("Products", "")),
                    "n":        int(row.get("N_Products", 0)),
                })
    return result

def get_kitchen_predicted(display_yw: str = "") -> dict:
    """从模型输出 CSV 获取厨房视图。"""
    df = load_csv(OUTPUT_DIR / "kitchen_view_v7.csv")
    if df is None:
        return {}
    yw = display_yw or get_predicted_week()
    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    df = df.sort_values(["_ord", "Qty"], ascending=[True, False])
    result = {}
    for day, ddf in df.groupby("Day"):
        result[day] = {
            "label": day_label(yw, day),
            "rows": [
                {"type": r["Type"], "product": r["Product"],
                 "qty": int(r["Qty"]), "range": r["Range"], "stores": int(r["Stores"])}
                for _, r in ddf.iterrows()
            ]
        }
    return result

def get_metrics():
    df = load_csv(OUTPUT_DIR / "metrics_v7.csv")
    if df is None:
        return None
    return df.iloc[0].to_dict()


# ── 历史实际数据（来自 1year.csv） ────────────────────────────
_raw_state  = {"df": None, "mtime": None}
_hist_cache = {}   # year_week → {"driver": {…}, "kitchen": {…}}

def _get_raw_df():
    """带 mtime 缓存的 1year.csv 加载器。文件变化时自动失效。
    内存优化：只加载必要列 + 使用 category 类型，内存占用从 ~400 MB 降至 ~80 MB。
    """
    _ensure_data_file()          # 若无 CSV 则尝试从 zip 解压
    if not DATA_FILE.exists():
        return None
    try:
        import pandas as pd
        mtime = DATA_FILE.stat().st_mtime
        if _raw_state["df"] is None or _raw_state["mtime"] != mtime:
            # ── 只读取需要的列，大幅减少内存 ──────────────────────
            needed = {"datum", "antal_ordrar", "antal_returer",
                      "ort", "namn", "sort", "typ"}
            # 先探测实际列名（兼容列名不一致的情况）
            header = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
            usecols = [c for c in header if c in needed]
            # category 类型大幅节省字符串列内存
            cat_cols = {"ort", "namn", "sort", "typ"}
            dtype = {c: "category" for c in usecols if c in cat_cols}
            df = pd.read_csv(DATA_FILE, usecols=usecols, dtype=dtype)
            df["datum"]   = pd.to_datetime(df["datum"], errors="coerce")
            df = df.dropna(subset=["datum"])
            df["_year"]   = df["datum"].dt.isocalendar().year.astype("int16")
            df["_week"]   = df["datum"].dt.isocalendar().week.astype("int8")
            df["_yw"]     = (df["_year"].astype(str) + "-W"
                             + df["_week"].astype(str).str.zfill(2)).astype("category")
            df["_day"]    = df["datum"].dt.day_name().astype("category")
            df["faktisk"] = (df["antal_ordrar"] - df["antal_returer"]).clip(lower=0).astype("int32")
            # 释放原始数值列节省内存
            df.drop(columns=["antal_ordrar", "antal_returer"], errors="ignore", inplace=True)
            # 过滤暂停产品
            mask = df["sort"].astype(str).str.contains(
                r"beställ ej|Paus|\(EC\)", case=False, na=False)
            df = df[~mask]
            # 过滤非门店
            store_sum  = df.groupby("namn")["faktisk"].sum()
            non_stores = store_sum[store_sum == 0].index.tolist() + ["Prover", "Alina Systems"]
            df = df[~df["namn"].isin(non_stores)]
            _raw_state["df"]    = df
            _raw_state["mtime"] = mtime
            _hist_cache.clear()
        return _raw_state["df"]
    except Exception:
        return None

def compute_week_actual(year_week: str):
    """
    从 1year.csv 计算某周的司机视图和厨房视图。
    返回 {"driver": {…}, "kitchen": {…}}，无数据时返回 None。
    """
    if year_week in _hist_cache:
        return _hist_cache[year_week]

    df = _get_raw_df()
    if df is None:
        return None

    week_df = df[df["_yw"] == year_week].copy()
    if week_df.empty:
        return None

    # ── 司机视图 ──
    sd = week_df.groupby(["ort", "namn", "_day"]).agg(
        qty=("faktisk", "sum")).reset_index()
    pd_ = (week_df[week_df["faktisk"] > 0]
           .groupby(["ort", "namn", "_day", "sort"])
           .agg(qty=("faktisk", "sum")).reset_index())

    driver = {}
    for _, row in sd[sd["qty"] > 0].iterrows():
        ort, namn, day, qty = row["ort"], row["namn"], row["_day"], int(row["qty"])
        prods = pd_[(pd_["ort"] == ort) & (pd_["namn"] == namn) &
                    (pd_["_day"] == day)]["sort"].tolist()
        driver.setdefault(ort, {}).setdefault(namn, []).append({
            "day":       day,
            "day_label": day_label(year_week, day),
            "qty":       qty,
            "range":     str(qty),   # 实际数据 = 精确值，无区间
            "products":  " | ".join(prods),
            "n":         len(prods),
        })
    for ort in driver:
        for namn in driver[ort]:
            driver[ort][namn].sort(key=lambda x: DAY_ORDER.get(x["day"], 9))

    # ── 厨房视图 ──
    cat = week_df.groupby(["_day", "typ", "sort"]).agg(
        qty=("faktisk", "sum"), stores=("namn", "nunique")).reset_index()
    cat = cat[cat["qty"] > 0]

    kitchen = {}
    for day, ddf in cat.groupby("_day"):
        ddf = ddf.sort_values("qty", ascending=False)
        kitchen[day] = {
            "label": day_label(year_week, day),
            "rows": [
                {"type": r["typ"], "product": r["sort"],
                 "qty": int(r["qty"]), "range": str(int(r["qty"])),
                 "stores": int(r["stores"])}
                for _, r in ddf.iterrows()
            ]
        }

    result = {"driver": driver, "kitchen": kitchen}
    _hist_cache[year_week] = result
    return result

def get_all_weeks():
    """返回所有可用周（实际 + 预测）。"""
    weeks = {}
    df = _get_raw_df()
    if df is not None:
        for yw in df["_yw"].unique():
            weeks[yw] = "actual"
    pred = get_predicted_week()
    if pred:
        # 模型输出周及之后都标为预测
        if pred not in weeks:
            weeks[pred] = "predicted"
        else:
            weeks[pred] = "predicted"   # 覆盖（最近几周数据不完整）
    return sorted([{"week": k, "type": v} for k, v in weeks.items()],
                  key=lambda x: x["week"])


# ── 路由 ──────────────────────────────────────────────────────
@app.route("/")
def index():
    has_output = (OUTPUT_DIR / "driver_view_v7.csv").exists()
    pred_week  = get_predicted_week() if has_output else ""
    metrics    = get_metrics() if has_output else None

    all_wks    = get_all_weeks()
    min_week   = all_wks[0]["week"] if all_wks else ""

    return render_template("index.html",
                           has_output=has_output,
                           pred_week=pred_week,
                           init_week=pred_week,
                           date_range=week_to_range(pred_week) if pred_week else "",
                           metrics=metrics,
                           min_week=min_week,
                           pw_required=bool(APP_PASSWORD))

@app.route("/api/weeks")
def api_weeks():
    return jsonify(get_all_weeks())

@app.route("/api/driver")
def api_driver():
    week = request.args.get("week", "").strip()
    pred = get_predicted_week()
    if not week:
        week = pred

    # 优先尝试历史实际数据
    actual = compute_week_actual(week) if week else None
    if actual is not None:
        return jsonify({
            "data":       actual["driver"],
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "actual",
        })

    # 无实际数据 → 返回模型预测（日期标签用请求的周）
    if pred:
        return jsonify({
            "data":       get_driver_predicted(display_yw=week),
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "predicted",
        })

    return jsonify({"data": {}, "year_week": week,
                    "date_range": week_to_range(week), "type": "nodata"})

@app.route("/api/kitchen")
def api_kitchen():
    week = request.args.get("week", "").strip()
    pred = get_predicted_week()
    if not week:
        week = pred

    actual = compute_week_actual(week) if week else None
    if actual is not None:
        return jsonify({
            "data":       actual["kitchen"],
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "actual",
        })

    if pred:
        return jsonify({
            "data":       get_kitchen_predicted(display_yw=week),
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "predicted",
        })

    return jsonify({"data": {}, "year_week": week,
                    "date_range": week_to_range(week), "type": "nodata"})

@app.route("/api/run", methods=["POST"])
def api_run():
    if not _password_ok():
        return jsonify({"ok": False, "msg": "密码错误"}), 403
    if pipeline_status["running"]:
        return jsonify({"ok": False, "msg": "流水线正在运行中"})
    threading.Thread(target=run_pipeline_thread, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify({
        "running": pipeline_status["running"],
        "done":    pipeline_status["done"],
        "error":   pipeline_status["error"],
        "log":     pipeline_status["log"][-200:],
    })

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if not _password_ok():
        return jsonify({"ok": False, "msg": "密码错误"}), 403
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "msg": "未选择文件"})
    if not f.filename.endswith(".csv"):
        return jsonify({"ok": False, "msg": "只接受 CSV 文件"})
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        f.save(str(DATA_FILE))
        # 使缓存失效
        _raw_state["df"] = None
        _hist_cache.clear()
        size_kb = DATA_FILE.stat().st_size // 1024
        return jsonify({"ok": True, "msg": f"上传成功 ({size_kb} KB)，可以运行训练了"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/diagnostics")
def api_diagnostics():
    path = OUTPUT_DIR / "diagnostics_v7.png"
    if path.exists():
        return send_file(str(path), mimetype="image/png")
    return "No diagnostics", 404

@app.route("/api/download/<name>")
def api_download(name):
    allowed = {"driver": "driver_view_v7.csv", "kitchen": "kitchen_view_v7.csv"}
    if name not in allowed:
        return "Not found", 404
    path = OUTPUT_DIR / allowed[name]
    if not path.exists():
        return "File not found", 404
    return send_file(str(path), as_attachment=True)


# ── 启动 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port     = int(os.environ.get("PORT", 5000))
    is_local = not os.environ.get("PORT")
    print("\n" + "=" * 50)
    print(f"  Eataway  http://127.0.0.1:{port}")
    if APP_PASSWORD:
        print("  密码保护: 已启用")
    print("=" * 50 + "\n")
    if is_local:
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)
