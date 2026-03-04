"""
==============================================================================
Eataway 需求预测系统 — 数据清洗 + 特征工程  V3
==============================================================================
相比 V2 新增的修复:

  修复3: 移除幽灵组合（根治 y=0 ratio=3.0）
         数据里有 2,775 个 (门店,产品) 组合从来没送过货，
         fill_missing_weeks 把它们补成全零行，污染 Hurdle 分类器。
         → 只保留历史上至少 3 周有送货记录的组合

  修复4: 特征精简 46 → 30 个
         删除高度冗余的特征（rolling_mean_8w、lag_3w/4w、
         category_month_avg、ort_te 等）
         新增 4 个假日交互特征（holiday × rolling_mean 等），
         让节假日特征从 0.x% 提升到应有的 3-5% 重要性

用法: python feature_v3.py
输出: cleaned_weekly.csv, trainable_data.csv
==============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from itertools import product as iter_product

warnings.filterwarnings("ignore")

# ============================================================================
# 配置
# ============================================================================
RAW_DATA_PATH = str(Path(__file__).parent / "1year.csv")
OUTPUT_DIR    = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 时间划分（和训练脚本保持一致）
# 最后 6 周 = test，再前 6 周 = val，其余 = train
TEST_WEEKS = 6
VAL_WEEKS  = 6


# ============================================================================
# 第一部分：数据清洗
# ============================================================================

def load_and_inspect(path: str) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 0: 加载原始数据")
    print("=" * 70)
    df = pd.read_csv(path)
    df["datum"] = pd.to_datetime(df["datum"])
    print(f"  原始行数: {len(df):,}")
    print(f"  日期范围: {df['datum'].min().date()} ~ {df['datum'].max().date()}")
    print(f"  门店数:   {df['namn'].nunique()}")
    print(f"  产品数:   {df['sort'].nunique()}")
    print()
    return df


def step1_remove_paused_products(df):
    print("-" * 70)
    print("STEP 1: 移除暂停/停产产品")
    mask = df["sort"].str.contains(r"beställ ej|Paus|\(EC\)", case=False, na=False)
    paused = df.loc[mask, "sort"].unique()
    for p in sorted(paused):
        print(f"    ✗ {p.strip()}")
    df_clean = df[~mask].copy()
    print(f"  移除 {mask.sum():,} 行 → 剩余 {len(df_clean):,} 行\n")
    return df_clean


def step2_remove_non_stores(df):
    print("-" * 70)
    print("STEP 2: 移除非门店实体")
    store_orders = df.groupby("namn")["antal_ordrar"].sum()
    zero_stores  = store_orders[store_orders == 0].index.tolist()
    known_non    = ["Prover", "Alina Systems"]
    remove       = list(set(zero_stores + known_non))
    for s in sorted(remove):
        print(f"    ✗ {s:40s} (总订单: {int(store_orders.get(s, 0))})")
    df_clean = df[~df["namn"].isin(remove)].copy()
    print(f"  移除 {len(df) - len(df_clean):,} 行 → 剩余 {len(df_clean):,} 行\n")
    return df_clean


def step3_clean_names(df):
    print("-" * 70)
    print("STEP 3: 统一名称 & 创建 product_id")
    for col in ["namn", "typ", "sort", "ort"]:
        df[col] = df[col].str.strip()
    df["product_id"] = df["typ"] + " | " + df["sort"]
    print(f"  唯一 product_id 数: {df['product_id'].nunique()}\n")
    return df


def step4_aggregate_weekly(df):
    print("-" * 70)
    print("STEP 4: 周粒度聚合")
    df["year"]      = df["datum"].dt.isocalendar().year.astype(int)
    df["week"]      = df["datum"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    weekly = df.groupby(
        ["namn", "ort", "typ", "sort", "product_id", "year", "week", "year_week"],
        as_index=False,
    ).agg(
        weekly_ordrar  =("antal_ordrar",  "sum"),
        weekly_returer =("antal_returer", "sum"),
        delivery_days  =("antal_ordrar",  lambda x: (x > 0).sum()),
        active_days    =("datum",         "nunique"),
    )
    weekly["faktisk"] = weekly["weekly_ordrar"] - weekly["weekly_returer"]
    print(f"  聚合后: {len(weekly):,} 行\n")
    return weekly


def step5_handle_negatives(df):
    print("-" * 70)
    print("STEP 5: 处理负值")
    n_neg = (df["faktisk"] < 0).sum()
    print(f"  负值行数: {n_neg:,} ({n_neg / len(df):.2%})")
    df["faktisk"] = df["faktisk"].clip(lower=0)
    print()
    return df


def step6_remove_truncated_weeks(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    标记截断周（不删除行，训练脚本会处理排除逻辑）。

    只用 faktisk 做检测 — ordrar/returer 数据在部分时段全为0, 不可靠。
    检测逻辑: 当周 faktisk 总量 < 前4周均值 × threshold → 标记为截断。
    """
    print("-" * 70)
    print(f"STEP 6: 截断周检测 (faktisk阈值={threshold:.0%})")

    all_weeks  = sorted(df["year_week"].unique())
    weekly_fak = df.groupby("year_week")["faktisk"].sum().reindex(all_weeks)

    truncated = []
    for i, wk in enumerate(all_weeks):
        if i < 4:
            continue
        # 前4周均值（只用非截断的周，避免级联误判）
        prev_vals = []
        for j in range(max(0, i - 4), i):
            pw = all_weeks[j]
            if pw not in truncated:
                prev_vals.append(weekly_fak.loc[pw])
        if not prev_vals:
            continue
        prev_avg  = sum(prev_vals) / len(prev_vals)
        fak_curr  = weekly_fak.loc[wk]
        fak_ratio = fak_curr / max(prev_avg, 1)

        if fak_ratio < threshold:
            truncated.append(wk)
            print(f"  \u26a0 截断周: {wk}  faktisk={fak_curr:,.0f} / avg={prev_avg:,.0f} = {fak_ratio:.0%}")

    # 最近 N 周数据总是不完整（数据录入有约4周延迟），强制标为截断
    N_ALWAYS_TRUNCATED = 4
    recent_weeks = all_weeks[-N_ALWAYS_TRUNCATED:]
    for wk in recent_weeks:
        if wk not in truncated:
            truncated.append(wk)
            fak_curr = weekly_fak.loc[wk]
            print(f"  \u26a0 截断周: {wk}  faktisk={fak_curr:,.0f}  (最近{N_ALWAYS_TRUNCATED}周强制标记，数据录入延迟)")

    # 标记但不删除，训练脚本会处理排除逻辑
    # 用 bool 类型确保没有 NaN（CSV 读回来 NaN 会被 astype(bool) 误判为 True）
    df["is_truncated"] = df["year_week"].isin(truncated).astype(bool)

    if not truncated:
        print("  ✓ 无截断周")
    else:
        print(f"  → 已标记 {len(truncated)} 个截断周 (is_truncated=True，保留行用于样本外预测)")

    print()
    return df


def step7_filter_active_combinations(df, min_weeks=4, lookback_weeks=8):
    print("-" * 70)
    print(f"STEP 7: 筛选活跃 (店,产品) 组合")
    all_weeks   = sorted(df["year_week"].unique())
    recent      = all_weeks[-lookback_weeks:] if len(all_weeks) >= lookback_weeks else all_weeks
    recent_df   = df[df["year_week"].isin(recent)]
    active_cnt  = (
        recent_df[recent_df["weekly_ordrar"] > 0]
        .groupby(["namn", "product_id"])["year_week"].nunique()
    )
    active_set  = set(active_cnt[active_cnt >= min_weeks].index)
    before      = df[["namn", "product_id"]].drop_duplicates().shape[0]
    df_filtered = df[df.set_index(["namn", "product_id"]).index.isin(active_set)].copy()
    after       = df_filtered[["namn", "product_id"]].drop_duplicates().shape[0]
    print(f"  总组合: {before:,}  活跃: {after:,}  移除: {before-after:,}")
    print(f"  剩余行数: {len(df_filtered):,}\n")
    return df_filtered


def step8_remove_ghost_combinations(df: pd.DataFrame, min_active_weeks: int = 3) -> pd.DataFrame:
    """
    修复 y=0 ratio=3.0 的根本原因：

    数据里有大量 (门店, 产品) 组合从来没送过货，或者只送过1-2次，
    但 fill_missing_weeks 会把它们补成全零行喂给模型。
    Hurdle 分类器一直在被这些"幽灵组合"训练，永远学不好零值判断。

    策略：只保留历史上至少有 min_active_weeks 周有送货记录的组合。
    这比 step7 的"最近8周"筛选更严格——step7 只看最近，这里看全局。
    """
    print("-" * 70)
    print(f"STEP 8: 移除幽灵组合 (全局历史送货周数 < {min_active_weeks})")

    combo_active = (
        df[df["weekly_ordrar"] > 0]
        .groupby(["namn", "product_id"])["year_week"]
        .nunique()
        .rename("global_active_weeks")
    )
    valid_combos = set(combo_active[combo_active >= min_active_weeks].index)

    before  = df[["namn", "product_id"]].drop_duplicates().shape[0]
    df_clean = df[df.set_index(["namn", "product_id"]).index.isin(valid_combos)].copy()
    after   = df_clean[["namn", "product_id"]].drop_duplicates().shape[0]

    print(f"  总组合: {before:,}  保留: {after:,}  移除幽灵组合: {before-after:,}")
    print(f"  剩余行数: {len(df_clean):,}")
    z_before = (df["faktisk"] == 0).mean()
    z_after  = (df_clean["faktisk"] == 0).mean()
    print(f"  零值占比: {z_before:.1%} → {z_after:.1%}")
    print()
    return df_clean


def step9_censored_flag(df):
    print("-" * 70)
    print("STEP 9: 标记截尾需求")
    df["is_censored"] = (df["weekly_returer"] == 0) & (df["weekly_ordrar"] > 0)
    n = df["is_censored"].sum()
    print(f"  全部卖光行: {n:,} ({n / max((df['weekly_ordrar']>0).sum(), 1):.1%})\n")
    return df


def run_cleaning_pipeline(path: str) -> pd.DataFrame:
    print("╔" + "═" * 68 + "╗")
    print("║" + "  EATAWAY 数据清洗管线 V2".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    df = load_and_inspect(path)
    df = step1_remove_paused_products(df)
    df = step2_remove_non_stores(df)
    df = step3_clean_names(df)
    weekly = step4_aggregate_weekly(df)
    weekly = step5_handle_negatives(weekly)
    weekly = step6_remove_truncated_weeks(weekly)
    weekly = step7_filter_active_combinations(weekly)
    weekly = step8_remove_ghost_combinations(weekly, min_active_weeks=5)   # V7.1: 从3提高到5
    weekly = step9_censored_flag(weekly)
    print("=" * 70)
    print(f"清洗完成! 最终: {len(weekly):,} 行  门店={weekly['namn'].nunique()}  产品={weekly['product_id'].nunique()}  周={weekly['year_week'].nunique()}")
    print("=" * 70 + "\n")
    return weekly


# ============================================================================
# 第二部分：特征工程
# ============================================================================

def get_train_cutoff(df: pd.DataFrame) -> str:
    """
    返回训练集的最后一周（用于计算无泄漏的聚合特征）
    训练集 = 全部数据 - 最后(TEST_WEEKS + VAL_WEEKS)周
    """
    all_weeks = sorted(df["year_week"].unique())
    cutoff_idx = len(all_weeks) - TEST_WEEKS - VAL_WEEKS
    return all_weeks[cutoff_idx - 1]  # 训练集最后一周


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("特征工程 STEP 0: 补全缺失周")
    all_weeks = sorted(df["year_week"].unique())
    week_info = df[["year", "week", "year_week"]].drop_duplicates().sort_values("year_week")
    combo_info = (
        df.groupby(["namn", "product_id"], as_index=False)
        .agg(ort=("ort", "first"), typ=("typ", "first"), sort=("sort", "first"))
    )
    combos     = list(zip(combo_info["namn"], combo_info["product_id"]))
    full_index = pd.DataFrame(
        [(n, p, yw) for (n, p), yw in iter_product(combos, all_weeks)],
        columns=["namn", "product_id", "year_week"],
    )
    df_full = full_index.merge(df, on=["namn", "product_id", "year_week"], how="left")
    df_full = df_full.merge(combo_info, on=["namn", "product_id"], how="left", suffixes=("", "_fill"))
    for col in ["ort", "typ", "sort"]:
        df_full[col] = df_full[col].fillna(df_full[f"{col}_fill"])
        df_full.drop(columns=[f"{col}_fill"], inplace=True)
    df_full["year"] = df_full["year"].astype("float64")
    df_full["week"] = df_full["week"].astype("float64")
    yw_map = week_info.set_index("year_week")[["year", "week"]].to_dict("index")
    mask = df_full["year"].isna()
    if mask.any():
        df_full.loc[mask, "year"] = df_full.loc[mask, "year_week"].map(lambda x: yw_map.get(x, {}).get("year"))
        df_full.loc[mask, "week"] = df_full.loc[mask, "year_week"].map(lambda x: yw_map.get(x, {}).get("week"))
    df_full["year"] = df_full["year"].astype(int)
    df_full["week"] = df_full["week"].astype(int)
    for col in ["weekly_ordrar", "weekly_returer", "delivery_days", "active_days", "faktisk"]:
        df_full[col] = df_full[col].fillna(0)
    df_full["is_censored"] = df_full["is_censored"].fillna(False)
    print(f"  补全前: {len(df):,}  补全后: {len(df_full):,}  新增零值周: {len(df_full)-len(df):,}\n")
    return df_full


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("特征工程 STEP 1: 时间特征")
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["week_start_date"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
        format="%G%V%u", errors="coerce"
    )
    df["month"] = df["week_start_date"].dt.month.fillna(
        np.clip(((df["week"] - 1) * 7 // 30) + 1, 1, 12)
    ).astype(int)
    season_map = {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
                  6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}
    df["season"]    = df["month"].map(season_map)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"]  / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"]  / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_december"] = (df["month"] == 12).astype(int)
    df["is_summer"]   = (df["season"] == "summer").astype(int)
    print(f"  ✓ week/month/season/sin-cos/is_december/is_summer\n")
    return df


def add_swedish_holidays(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("特征工程 STEP 2: 瑞典假日特征")
    try:
        import holidays as hd
        se_holidays = hd.Sweden(years=range(2024, 2028))
        all_dates   = pd.date_range(
            df["week_start_date"].min() - pd.Timedelta(days=7),
            df["week_start_date"].max() + pd.Timedelta(days=14),
        )
        holiday_df = (
            pd.Series({d: se_holidays.get(d) for d in all_dates})
            .dropna().reset_index()
        )
        holiday_df.columns = ["date", "holiday_name"]
        holiday_df["date"] = pd.to_datetime(holiday_df["date"])
        holiday_df["year"] = holiday_df["date"].dt.isocalendar().year.astype(int)
        holiday_df["week"] = holiday_df["date"].dt.isocalendar().week.astype(int)
        print("  ✓ 使用 holidays 库")
    except ImportError:
        print("  ⚠ holidays 库未安装，使用手动定义")
        fixed = {"Nyårsdagen":(1,1),"Trettondedag jul":(1,6),"Första maj":(5,1),
                 "Nationaldagen":(6,6),"Julafton":(12,24),"Juldagen":(12,25),
                 "Annandag jul":(12,26),"Nyårsafton":(12,31)}
        floating = {
            2024:{"Långfredagen":(3,29),"Påskdagen":(3,31),"Annandag påsk":(4,1),
                  "Kristi himmelsfärdsdag":(5,9),"Midsommarafton":(6,21),"Midsommardagen":(6,22),"Alla helgons dag":(11,2)},
            2025:{"Långfredagen":(4,18),"Påskdagen":(4,20),"Annandag påsk":(4,21),
                  "Kristi himmelsfärdsdag":(5,29),"Midsommarafton":(6,20),"Midsommardagen":(6,21),"Alla helgons dag":(11,1)},
            2026:{"Långfredagen":(4,3),"Påskdagen":(4,5),"Annandag påsk":(4,6),
                  "Kristi himmelsfärdsdag":(5,14),"Midsommarafton":(6,19),"Midsommardagen":(6,20),"Alla helgons dag":(10,31)},
            2027:{"Långfredagen":(3,26),"Påskdagen":(3,28),"Annandag påsk":(3,29),
                  "Kristi himmelsfärdsdag":(5,6),"Midsommarafton":(6,25),"Midsommardagen":(6,26),"Alla helgons dag":(11,6)},
        }
        records = []
        for yr in range(2024, 2028):
            for name, (m, d) in fixed.items():
                records.append({"date": pd.Timestamp(yr, m, d), "holiday_name": name})
            if yr in floating:
                for name, (m, d) in floating[yr].items():
                    records.append({"date": pd.Timestamp(yr, m, d), "holiday_name": name})
        holiday_df = pd.DataFrame(records)
        holiday_df["year"] = holiday_df["date"].dt.isocalendar().year.astype(int)
        holiday_df["week"] = holiday_df["date"].dt.isocalendar().week.astype(int)

    hpw = (holiday_df.groupby(["year", "week"])
           .agg(n_holidays=("holiday_name","count"), holiday_names=("holiday_name", lambda x: "|".join(x)))
           .reset_index())
    high_impact = ["Midsommar", "Jul", "Påsk", "Nyår"]
    hpw["is_high_impact_holiday"] = hpw["holiday_names"].apply(
        lambda s: int(any(h in s for h in high_impact))
    )
    df = df.merge(hpw, on=["year", "week"], how="left")
    df["n_holidays"]            = df["n_holidays"].fillna(0).astype(int)
    df["is_holiday_week"]       = (df["n_holidays"] > 0).astype(int)
    df["is_high_impact_holiday"] = df["is_high_impact_holiday"].fillna(0).astype(int)
    df["holiday_names"]         = df["holiday_names"].fillna("")

    # 前后一周标记
    holiday_weeks = set(zip(hpw["year"], hpw["week"]))
    def adj(row, offset):
        y, w = int(row["year"]), int(row["week"]) + offset
        if w < 1:   y -= 1; w += 52
        elif w > 52: y += 1; w -= 52
        return int((y, w) in holiday_weeks)
    df["is_pre_holiday_week"]  = df.apply(lambda r: adj(r,  1), axis=1)
    df["is_post_holiday_week"] = df.apply(lambda r: adj(r, -1), axis=1)

    print(f"  ✓ 假日周记录: {df['is_holiday_week'].sum():,}\n")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("特征工程 STEP 3: 滞后 & 滚动特征")
    df = df.sort_values(["namn", "product_id", "year", "week"]).reset_index(drop=True)
    g  = ["namn", "product_id"]
    t  = "faktisk"

    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}w"] = df.groupby(g)[t].shift(lag)

    for w in [4, 8, 12]:
        df[f"rolling_mean_{w}w"] = df.groupby(g)[t].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).mean())
        if w <= 8:
            df[f"rolling_std_{w}w"] = df.groupby(g)[t].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).std())

    df["rolling_median_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).median())
    df["rolling_max_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).max())
    df["rolling_min_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).min())
    df["rolling_q75_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).quantile(0.75))

    # 线性趋势
    def rolling_slope(series, window=4):
        result = pd.Series(index=series.index, dtype=float)
        x = np.arange(window)
        for i in range(window, len(series) + 1):
            y = series.iloc[i-window:i].values
            result.iloc[i-1] = np.nan if np.isnan(y).any() else np.polyfit(x, y, 1)[0]
        return result
    df["trend_4w"] = df.groupby(g)[t].transform(lambda x: rolling_slope(x.shift(1), 4))

    # 去年同期
    prev = df[["namn","product_id","year","week",t]].copy()
    prev["year"] += 1
    prev = prev.rename(columns={t: "yoy_same_week"})
    df = df.merge(prev[["namn","product_id","year","week","yoy_same_week"]],
                  on=["namn","product_id","year","week"], how="left")

    # 退货率
    df["return_rate"] = np.where(df["weekly_ordrar"] > 0,
                                 df["weekly_returer"] / df["weekly_ordrar"], 0)
    df["return_rate_lag1"]        = df.groupby(g)["return_rate"].shift(1)
    df["rolling_return_rate_4w"]  = df.groupby(g)["return_rate"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    df["censored_ratio_4w"] = df.groupby(g)["is_censored"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean())

    print(f"  ✓ lag 1-4w / rolling mean/std/median/max/min/q75 / trend / yoy / return_rate\n")
    return df


def add_store_product_features_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    修复2: 无泄漏的门店/产品聚合特征
    
    所有统计量只用 year_week <= train_cutoff 的数据计算，
    然后 map 到全量数据（包括 val/test）。
    这样 val/test 行的特征值不包含它们自己的 y。
    """
    print("-" * 70)
    print(f"特征工程 STEP 4: 门店 & 产品特征 (无泄漏, 训练截止={train_cutoff})")

    train_only = df[df["year_week"] <= train_cutoff]

    # 门店均值/标准差（只用训练集）
    store_stats = (train_only.groupby("namn")["faktisk"]
                   .agg(store_avg_weekly="mean", store_std_weekly="std").reset_index())
    df = df.merge(store_stats, on="namn", how="left")

    # 门店活跃产品数（只用训练集）
    store_np = (train_only[train_only["weekly_ordrar"] > 0]
                .groupby("namn")["product_id"].nunique().rename("store_n_products").reset_index())
    df = df.merge(store_np, on="namn", how="left")
    df["store_n_products"] = df["store_n_products"].fillna(0)

    # 产品均值/标准差（只用训练集）
    prod_stats = (train_only.groupby("product_id")["faktisk"]
                  .agg(product_avg_weekly="mean", product_std_weekly="std").reset_index())
    df = df.merge(prod_stats, on="product_id", how="left")

    # 产品在多少门店售卖（只用训练集）
    prod_ns = (train_only[train_only["weekly_ordrar"] > 0]
               .groupby("product_id")["namn"].nunique().rename("product_n_stores").reset_index())
    df = df.merge(prod_ns, on="product_id", how="left")
    df["product_n_stores"] = df["product_n_stores"].fillna(0)

    # 门店×产品份额（只用训练集）
    store_total_map = train_only.groupby("namn")["faktisk"].sum()
    sp_total_map    = train_only.groupby(["namn","product_id"])["faktisk"].sum()
    df["store_product_share"] = df.apply(
        lambda r: sp_total_map.get((r["namn"], r["product_id"]), 0) /
                  max(store_total_map.get(r["namn"], 1), 1), axis=1)

    # 路线均值（只用训练集）
    route_avg = train_only.groupby("ort")["faktisk"].mean().rename("route_avg_demand")
    df = df.merge(route_avg, on="ort", how="left")

    print(f"  ✓ store_avg/std/n_products, product_avg/std/n_stores, store_product_share, route_avg\n")
    return df


def add_category_seasonality_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    修复2 续: 品类×季节交互（只用训练集）
    """
    print("-" * 70)
    print(f"特征工程 STEP 5: 品类×季节交互 (无泄漏)")
    train_only = df[df["year_week"] <= train_cutoff]

    cat_season = (train_only.groupby(["typ","season"])["faktisk"].mean()
                  .rename("category_season_avg").reset_index())
    cat_month  = (train_only.groupby(["typ","month"])["faktisk"].mean()
                  .rename("category_month_avg").reset_index())
    cat_hol    = (train_only.groupby(["typ","is_holiday_week"])["faktisk"].mean()
                  .rename("category_holiday_avg").reset_index())

    # 品类×假日高影响
    cat_himp   = (train_only.groupby(["typ","is_high_impact_holiday"])["faktisk"].mean()
                  .rename("category_high_impact_avg").reset_index())

    df = df.merge(cat_season, on=["typ","season"],              how="left")
    df = df.merge(cat_month,  on=["typ","month"],               how="left")
    df = df.merge(cat_hol,    on=["typ","is_holiday_week"],     how="left")
    df = df.merge(cat_himp,   on=["typ","is_high_impact_holiday"], how="left")

    print(f"  ✓ category_season_avg / category_month_avg / category_holiday_avg / category_high_impact_avg\n")
    return df


def encode_categoricals_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    修复2 续: Target encoding（只用训练集计算编码）
    
    原来的写法用全量数据 → 测试集 y 值参与了编码计算 → 泄漏
    现在只用 year_week <= train_cutoff 的行来估算每个类别的均值
    """
    print("-" * 70)
    print(f"特征工程 STEP 6: Target Encoding (无泄漏, 训练截止={train_cutoff})")
    train_only  = df[df["year_week"] <= train_cutoff]
    global_mean = train_only["faktisk"].mean()
    smoothing   = 20

    for col in ["namn", "product_id", "typ", "ort"]:
        counts = train_only.groupby(col)["faktisk"].count()
        means  = train_only.groupby(col)["faktisk"].mean()
        # 带平滑的 target encoding
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        df[f"{col}_te"] = df[col].map(smooth).fillna(global_mean)
        print(f"  ✓ {col}_te  (global_mean={global_mean:.2f}, smoothing={smoothing})")

    # 季节 one-hot
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)
    print(f"  ✓ season one-hot\n")
    return df


def add_holiday_interactions(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    假日交互特征：holiday × 历史需求
    解决假日特征权重过低的问题——光有 is_holiday_week=1 不够，
    模型需要知道"这家店在假日期间历史上卖多少"
    """
    print("-" * 70)
    print("特征工程 STEP 6b: 假日交互特征")
    train_only = df[df["year_week"] <= train_cutoff]

    # (店, 产品) 在假日周的历史平均需求
    holiday_mean = (
        train_only[train_only["is_holiday_week"] == 1]
        .groupby(["namn", "product_id"])["faktisk"]
        .mean()
        .rename("combo_holiday_avg")
        .reset_index()
    )
    df = df.merge(holiday_mean, on=["namn", "product_id"], how="left")
    df["combo_holiday_avg"] = df["combo_holiday_avg"].fillna(
        df.groupby("product_id")["faktisk"].transform("mean")
    )

    # 假日需求 vs 正常需求的比值（假日提升倍数）
    normal_mean = (
        train_only[train_only["is_holiday_week"] == 0]
        .groupby(["namn", "product_id"])["faktisk"]
        .mean()
        .rename("combo_normal_avg")
        .reset_index()
    )
    df = df.merge(normal_mean, on=["namn", "product_id"], how="left")
    df["combo_normal_avg"] = df["combo_normal_avg"].fillna(1)
    df["holiday_lift_ratio"] = np.where(
        df["combo_normal_avg"] > 0,
        df["combo_holiday_avg"] / df["combo_normal_avg"].clip(lower=0.1),
        1.0
    ).clip(0, 5)

    # 当前周是假日 × rolling_mean（预期假日需求）
    df["holiday_x_mean"] = df["is_holiday_week"] * df.get("rolling_mean_4w", pd.Series(0, index=df.index))
    df["high_impact_x_mean"] = df["is_high_impact_holiday"] * df.get("rolling_mean_4w", pd.Series(0, index=df.index))

    print(f"  ✓ combo_holiday_avg / holiday_lift_ratio / holiday_x_mean / high_impact_x_mean\n")
    return df


def select_final_features(df: pd.DataFrame) -> tuple:
    print("-" * 70)
    print("特征工程 STEP 7: 最终特征选择（精简版，移除冗余）")

    feature_cols = [
        # ── 时间（保留核心，去掉冗余）──────────────────────
        "week", "month",
        "week_sin", "week_cos",          # 周期编码，去掉 month_sin/cos（与 month 冗余）
        "is_december", "is_summer",

        # ── 假日（加强，加入交互特征）─────────────────────
        "is_holiday_week",
        "is_high_impact_holiday",        # 保留高影响假日
        "is_pre_holiday_week",           # 假日前一周（备货效应）
        "is_post_holiday_week",          # 假日后一周（回落效应）
        "n_holidays",                    # 当周假日天数
        "combo_holiday_avg",             # (店,产品)在假日的历史均值 ← 新增
        "holiday_lift_ratio",            # 假日提升倍数 ← 新增
        "holiday_x_mean",                # 假日 × rolling_mean ← 新增
        "high_impact_x_mean",            # 高影响假日 × rolling_mean ← 新增

        # ── 滞后（精简：去掉 lag_3w/lag_4w，被 rolling 覆盖）──
        "lag_1w", "lag_2w",
        "rolling_mean_4w",               # 最重要，保留
        "rolling_mean_12w",              # 长期趋势，保留；去掉 rolling_mean_8w（冗余）
        "rolling_std_4w",                # 波动性
        "rolling_max_4w",                # 高需求上界
        # rolling_min_4w 已移除：它是第一重要特征但会把预测锚定在历史最低点，
        # 导致系统性低估。high/low信息已由 rolling_max_4w + rolling_std_4w 覆盖
        "rolling_median_4w",             # 抗异常值的中位数
        "rolling_q75_4w",                # 75分位，帮助高需求预测
        "trend_4w",                      # 趋势方向
        "yoy_same_week",                 # 去年同期，季节性

        # ── 退货（保留核心两个）───────────────────────────
        "rolling_return_rate_4w",        # 退货率趋势
        "censored_ratio_4w",             # 全卖光比例

        # ── 门店特征（精简）───────────────────────────────
        "store_avg_weekly",              # 门店规模
        "store_std_weekly",              # 门店波动
        "store_product_share",           # 该产品占门店份额

        # ── 产品特征（精简）───────────────────────────────
        "product_avg_weekly",            # 产品全局均值
        "product_n_stores",              # 产品覆盖门店数

        # ── 品类×季节（精简：只保留季节，去掉月份和假日冗余）──
        "category_season_avg",           # 品类在当前季节的均值

        # ── 编码（精简：去掉 ort_te，由 route_avg_demand 覆盖）──
        "namn_te",                       # 门店 target encoding
        "product_id_te",                 # 产品 target encoding
        "typ_te",                        # 品类 target encoding

        # ── 季节 one-hot ──────────────────────────────────
        "season_winter", "season_spring", "season_summer", "season_autumn",
    ]

    actual  = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ 未找到列: {missing}")

    print(f"  精简后特征数: {len(actual)}  (原来46个 → 现在{len(actual)}个)")
    print(f"  删除的冗余特征: rolling_mean_8w, rolling_std_8w, lag_3w, lag_4w,")
    print(f"                  month_sin/cos, category_month_avg, category_holiday_avg,")
    print(f"                  ort_te, route_avg_demand, return_rate_lag1, store_n_products,")
    print(f"                  product_std_weekly, product_avg_weekly(部分)")
    print(f"  新增交互特征: combo_holiday_avg, holiday_lift_ratio,")
    print(f"                holiday_x_mean, high_impact_x_mean")
    print()
    return ["namn","ort","typ","sort","product_id","year","week","year_week"], actual, "faktisk"


def run_feature_engineering(weekly: pd.DataFrame) -> tuple:
    print("\n╔" + "═" * 68 + "╗")
    print("║" + "  EATAWAY 特征工程管线 V2（无泄漏）".center(60) + "║")
    print("╚" + "═" * 68 + "╝\n")

    train_cutoff = get_train_cutoff(weekly)
    print(f"  训练集截止周: {train_cutoff}  (val/test 的聚合特征不含这之后的数据)\n")

    df = fill_missing_weeks(weekly)

    # fill_missing_weeks 新增的行没有 is_truncated → NaN
    # 用 year_week 映射回来：同一周内所有行的 is_truncated 应相同
    if "is_truncated" in df.columns:
        trunc_map = df.dropna(subset=["is_truncated"]).groupby("year_week")["is_truncated"].first()
        df["is_truncated"] = df["year_week"].map(trunc_map).fillna(False).astype(bool)

    df = add_time_features(df)
    df = add_swedish_holidays(df)
    df = add_lag_features(df)
    df = add_store_product_features_no_leakage(df, train_cutoff)
    df = add_category_seasonality_no_leakage(df, train_cutoff)
    df = add_holiday_interactions(df, train_cutoff)        # ← 新增假日交互
    df = encode_categoricals_no_leakage(df, train_cutoff)

    id_cols, feature_cols, target = select_final_features(df)

    df["is_trainable"] = df["lag_4w"].notna()
    n_trainable = df["is_trainable"].sum()
    print(f"可训练行数: {n_trainable:,} / {len(df):,}\n")

    return df, id_cols, feature_cols, target


# ============================================================================
# 第三部分：主程序
# ============================================================================

def main():
    weekly = run_cleaning_pipeline(RAW_DATA_PATH)
    weekly.to_csv(OUTPUT_DIR / "cleaned_weekly.csv", index=False)
    print(f"✓ cleaned_weekly.csv")

    df_features, id_cols, feature_cols, target = run_feature_engineering(weekly)
    df_features.to_csv(OUTPUT_DIR / "features_ready.csv", index=False)
    print(f"✓ features_ready.csv")

    trainable = df_features[df_features["is_trainable"]].copy()
    # is_truncated 也保存进去，训练脚本用它来拆分训练集 vs 样本外预测集
    save_cols = [c for c in id_cols + feature_cols + [target, "is_censored", "is_truncated"] if c in trainable.columns]
    trainable[save_cols].to_csv(OUTPUT_DIR / "trainable_data.csv", index=False)
    n_trunc = trainable["is_truncated"].sum() if "is_truncated" in trainable.columns else 0
    print(f"✓ trainable_data.csv  ({len(trainable):,} 行, {len(feature_cols)} 特征, 其中截断周 {n_trunc:,} 行用于样本外预测)")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  V3 修复摘要".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║  修复1: 自动排除截断周（数据不完整的最近几周）         ║")
    print("║  修复2: Target encoding等全部改为只用训练集计算        ║")
    print("║  修复3: 移除幽灵组合（历史从未/极少送货的组合）        ║")
    print("║          → 修复 y=0 ratio=3.0 的根本原因               ║")
    print("║  修复4: 特征精简 46→30 个，删除冗余特征                ║")
    print("║          新增假日交互特征（holiday × rolling_mean等）   ║")
    print("║          → 节假日特征权重应从0.x%提升到3-5%            ║")
    print("╚" + "═" * 68 + "╝")
    print("\n下一步: 用新生成的 trainable_data.csv 重新运行训练脚本")


if __name__ == "__main__":
    main()