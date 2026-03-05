"""
==============================================================================
Eataway V4 — 校准栅栏模型 + 偏差校正 + 天气特征
==============================================================================
V3 遗留问题 & V4 修复:

  问题1: y=0 比率=175 (模型几乎不预测零)
    根因: V3 软融合 P*reg 永远不输出零
    修复: 校准栅栏 — P_cal < 阈值 → 0, P_cal >= 阈值 → reg_pred (不乘P!)

  问题2: y=6-10 比率=0.81, y=11+ 比率=0.88 (高需求低估)
    根因: V3 P*reg 缩小预测; log空间压缩
    修复: 栅栏不乘P + 分层偏差校正因子

  问题3: 信息量瓶颈
    修复: 天气特征 (Open-Meteo/SMHI)

用法:
  1. 先运行 eataway_weather.py 获取天气数据 (可选)
  2. python eataway_train_v4.py

输入: trainable_data.csv + weather_weekly.csv(可选)
输出: D:/eataway_output_v4/
==============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import json
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ============================================================================
# 配置
OUTPUT_DIR = Path(__file__).parent / "output_v7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)# ============================================================================
DATA_PATH    = Path(__file__).parent / "trainable_data.csv"
WEATHER_PATH = str(Path(__file__).parent / "weather_weekly.csv")


SEED = 42

TARGET = "faktisk"
ID_COLS = ["namn", "ort", "typ", "sort", "product_id", "year", "week", "year_week"]

BASE_FEATURES = [
    "week", "month", "week_sin", "week_cos", "month_sin", "month_cos",
    "is_december", "is_summer",
    "is_holiday_week", "n_holidays", "is_high_impact_holiday",
    "is_pre_holiday_week", "is_post_holiday_week",
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "rolling_mean_4w", "rolling_mean_8w", "rolling_mean_12w",
    "rolling_std_4w", "rolling_std_8w",
    "rolling_median_4w", "rolling_max_4w",
    # rolling_min_4w 已移除：会把预测锚定在历史最低点，导致系统性低估
    "trend_4w", "yoy_same_week",
    "return_rate_lag1", "rolling_return_rate_4w", "censored_ratio_4w",
    "store_avg_weekly", "store_std_weekly", "store_n_products",
    "store_product_share", "route_avg_demand",
    "product_avg_weekly", "product_std_weekly", "product_n_stores",
    "category_season_avg", "category_month_avg", "category_holiday_avg",
    "namn_te", "product_id_te", "typ_te", "ort_te",
    "season_winter", "season_spring", "season_summer", "season_autumn",
]

ORT_TO_CITY = {
    "Gävle": "Gävle",
    "Stockholm Mellan": "Stockholm", "Stockholm N": "Stockholm",
    "Stockholm Södra": "Stockholm", "Stockholm Väst": "Stockholm",
    "Stockholm Xtra": "Stockholm", "Stockholm Östra": "Stockholm",
    "Uppsala Rutt": "Uppsala", "Västerås": "Västerås",
}


# ============================================================================
# P0: 不完整周检测
# ============================================================================
def detect_incomplete_weeks(df, threshold=0.7):
    print("=" * 70)
    print("P0: 检测不完整数据周")
    print("=" * 70)
    weekly = df[df[TARGET] > 0].groupby("year_week").agg(
        stores=("namn", "nunique"), demand=(TARGET, "sum"))
    all_wks = sorted(df["year_week"].unique())
    weekly = weekly.reindex(all_wks)
    bad = []
    for i, wk in enumerate(all_wks):
        if i < 4:
            continue
        c = weekly.loc[wk]
        p = weekly.iloc[max(0, i-4):i]
        sr = c["stores"] / max(p["stores"].mean(), 1)
        dr = c["demand"] / max(p["demand"].mean(), 1)
        if sr < threshold or dr < threshold:
            bad.append(wk)
            print(f"  Warning {wk}: store_ratio={sr:.0%} demand_ratio={dr:.0%}")
    if not bad:
        print("  OK no incomplete weeks")
    print()
    return bad


# ============================================================================
# 天气特征
# ============================================================================
def merge_weather(df):
    print("=" * 70)
    print("天气特征合并")
    print("=" * 70)

    wp = Path(WEATHER_PATH)
    if not wp.exists():
        print(f"  Warning: {WEATHER_PATH} not found")
        print("  -> Skip weather (run eataway_weather.py first)")
        print()
        return df, []

    weather = pd.read_csv(wp)
    print(f"  天气: {len(weather)} rows, {weather['city'].nunique()} cities")
    print(f"  范围: {weather['year_week'].min()} ~ {weather['year_week'].max()}")

    df["weather_city"] = df["ort"].map(ORT_TO_CITY).fillna("Stockholm")

    weather_cols = [
        "temp_mean", "temp_max", "temp_min", "temp_range",
        "precip_total", "rain_total", "snow_total",
        "wind_max", "gust_max", "sunshine_hrs",
        "rain_days", "snow_days",
        "is_rainy_week", "is_snowy_week", "is_cold", "is_hot",
    ]
    avail = [c for c in weather_cols if c in weather.columns]

    wr = weather.rename(columns={"city": "weather_city"})[["weather_city", "year_week"] + avail]
    before = len(df)
    df = df.merge(wr, on=["weather_city", "year_week"], how="left")
    assert len(df) == before

    matched = df[avail[0]].notna().sum()
    print(f"  匹配率: {matched}/{len(df)} ({matched/len(df):.1%})")

    for c in avail:
        df[c] = df[c].fillna(df[c].median())

    extra = []

    # 温度异常 (偏离月均多少度)
    if "temp_mean" in df.columns and "month" in df.columns:
        mm = df.groupby("month")["temp_mean"].transform("mean")
        df["temp_anomaly"] = df["temp_mean"] - mm
        extra.append("temp_anomaly")

    # 恶劣天气综合评分
    if "precip_total" in df.columns and "wind_max" in df.columns:
        df["bad_weather"] = (
            (df["precip_total"] > df["precip_total"].quantile(0.75)).astype(int) +
            (df["wind_max"] > df["wind_max"].quantile(0.75)).astype(int) +
            df.get("is_snowy_week", pd.Series(0, index=df.index)).astype(int)
        )
        extra.append("bad_weather")

    # 上周天气 (滞后)
    for col in ["temp_mean", "precip_total"]:
        if col in df.columns:
            lc = f"{col}_lag1w"
            df[lc] = df.groupby(["namn", "product_id"])[col].shift(1)
            df[lc] = df[lc].fillna(df[col].median())
            extra.append(lc)

    all_wf = avail + extra
    print(f"  天气特征: {len(all_wf)} 个")
    print()
    return df, all_wf


# ============================================================================
# 需求特征
# ============================================================================
def add_demand_features(df):
    print("=" * 70)
    print("需求特征工程")
    print("=" * 70)
    df = df.sort_values(["namn", "product_id", "year", "week"]).reset_index(drop=True)
    g = ["namn", "product_id"]
    new = []

    df["recent_zero_count"] = df.groupby(g)[TARGET].transform(
        lambda x: (x.shift(1) == 0).rolling(4, min_periods=1).sum())
    new.append("recent_zero_count")

    def wsn(x):
        r = pd.Series(index=x.index, dtype=float)
        last = -1
        for i in range(len(x)):
            if i > 0 and x.iloc[i-1] > 0:
                last = i - 1
            r.iloc[i] = (i - last) if last >= 0 else 8
        return r

    df["weeks_since_nonzero"] = df.groupby(g)[TARGET].transform(wsn).clip(0, 12)
    new.append("weeks_since_nonzero")

    df["active_ratio_8w"] = df.groupby(g)[TARGET].transform(
        lambda x: (x.shift(1) > 0).rolling(8, min_periods=2).mean()).fillna(0.5)
    new.append("active_ratio_8w")

    def nz_mean(x, w=4):
        r = pd.Series(index=x.index, dtype=float)
        s = x.shift(1)
        for i in range(len(x)):
            start = max(0, i-w)
            v = s.iloc[start:i]
            nz = v[v > 0]
            r.iloc[i] = nz.mean() if len(nz) > 0 else 0
        return r

    df["nonzero_mean_4w"] = df.groupby(g)[TARGET].transform(lambda x: nz_mean(x, 4))
    new.append("nonzero_mean_4w")

    rm4 = df.get("rolling_mean_4w", pd.Series(0, index=df.index))
    rs4 = df.get("rolling_std_4w", pd.Series(0, index=df.index))
    df["demand_cv_4w"] = np.where(rm4 > 0, rs4 / rm4.clip(lower=0.1), 0)
    df["demand_cv_4w"] = df["demand_cv_4w"].clip(0, 5).fillna(0)
    new.append("demand_cv_4w")

    df["lag1_zscore"] = np.where(
        rs4 > 0,
        (df.get("lag_1w", pd.Series(0, index=df.index)) - rm4) / rs4.clip(lower=0.1), 0)
    df["lag1_zscore"] = pd.to_numeric(df["lag1_zscore"], errors="coerce").fillna(0).clip(-5, 5)
    new.append("lag1_zscore")

    df["is_positive"] = (df[TARGET] > 0).astype(int)
    df["log_target"] = np.log1p(df[TARGET])

    print(f"  需求特征: {len(new)} 个")
    print()
    return df, new


# ============================================================================
# 加载
# ============================================================================
def load_and_prepare():
    print("\n" + "=" * 70)
    print("  EATAWAY V4 — 数据准备")
    print("=" * 70 + "\n")

    df_all = pd.read_csv(DATA_PATH)
    print(f"  原始: {len(df_all):,} 行\n")

    # 优先用 feature.py 标记的 is_truncated 列；若没有则自己检测
    if "is_truncated" in df_all.columns:
        # CSV 读回来可能是 str/float/NaN，统一转成 bool
        df_all["is_truncated"] = (
            df_all["is_truncated"]
            .fillna(False)
            .apply(lambda x: str(x).strip().lower() in ("true", "1", "1.0"))
        )
        truncated_weeks = df_all[df_all["is_truncated"]]["year_week"].unique().tolist()
        print(f"  截断周 (来自 feature.py): {truncated_weeks}")
        df = df_all[~df_all["is_truncated"]].copy()
    else:
        bad = detect_incomplete_weeks(df_all)
        truncated_weeks = bad
        df = df_all[~df_all["year_week"].isin(bad)].copy() if bad else df_all.copy()

    # 把截断周单独保存，训练后用于样本外预测
    df_holdout = df_all[df_all["year_week"].isin(truncated_weeks)].copy() if truncated_weeks else pd.DataFrame()
    print(f"  训练可用: {len(df):,} 行 | 样本外预测(截断周): {len(df_holdout):,} 行\n")

    df, wf = merge_weather(df)
    df, df_feats = add_demand_features(df)

    # holdout 也加同样的需求特征（用训练集学到的统计量填充）
    if not df_holdout.empty:
        df_holdout, _ = merge_weather(df_holdout)
        df_holdout, _ = add_demand_features(df_holdout)

    features = list(dict.fromkeys(
        [c for c in BASE_FEATURES + df_feats + wf if c in df.columns]))
    print(f"  最终: {len(df):,} 行, {len(features)} 特征\n")
    return df, features, df_holdout


def time_split(df, test_w=6, val_w=6):
    print("=" * 70)
    print("时间划分")
    print("=" * 70)
    wks = (df[["year","week","year_week"]].drop_duplicates()
           .sort_values(["year","week"]).reset_index(drop=True))
    n = len(wks)
    ts, vs = n - test_w, n - test_w - val_w
    tw = wks.iloc[:vs]["year_week"].tolist()
    vw = wks.iloc[vs:ts]["year_week"].tolist()
    tew = wks.iloc[ts:]["year_week"].tolist()
    dtr = df[df["year_week"].isin(tw)].copy()
    dva = df[df["year_week"].isin(vw)].copy()
    dte = df[df["year_week"].isin(tew)].copy()
    for nm, d, w in [("Train",dtr,tw),("Val",dva,vw),("Test",dte,tew)]:
        print(f"  {nm}: {len(d):>8,} rows | {len(w)} weeks | {w[0]}~{w[-1]}")
    print()
    return dtr, dva, dte


# ============================================================================
# V4 核心: CalibratedHurdleModel
# ============================================================================

class CalibratedHurdleModel:
    """
    V4 核心

    V2: 硬阈值0.45 + 未校准P → FN=886
    V3: 软融合P*reg → y=0比率=175, 高需求缩水
    V4: 校准栅栏 + 不乘P + 分层偏差校正

    关键: P_cal >= threshold → reg_pred (完整值, 不乘P!)
    """

    def __init__(self):
        self.cls_model = None
        self.reg_model = None
        self.calibrator = None
        self.threshold = 0.30
        self.bias_factors = None

    def fit(self, df_train, df_val, features):
        X_tr, X_va = df_train[features], df_val[features]

        # Stage 1: 分类
        print("  -- Stage 1: Classifier --")
        y1t = df_train["is_positive"]
        y1v = df_val["is_positive"]
        p1 = {"objective":"binary","metric":"binary_logloss",
              "learning_rate":0.05,"num_leaves":31,"max_depth":6,
              "min_child_samples":30,"subsample":0.8,"colsample_bytree":0.8,
              "reg_alpha":0.5,"reg_lambda":2.0,"is_unbalance":True,
              "n_jobs":-1,"seed":SEED,"verbose":-1}
        d1 = lgb.Dataset(X_tr, label=y1t)
        d1v = lgb.Dataset(X_va, label=y1v, reference=d1)
        self.cls_model = lgb.train(
            p1, d1, num_boost_round=3000, valid_sets=[d1v],
            callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])

        # Isotonic 校准
        pr = self.cls_model.predict(X_va)
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.calibrator.fit(pr, y1v.values)
        pc = self.calibrator.predict(pr)

        print(f"    best_iter={self.cls_model.best_iteration}")
        low_mask = pr < 0.1
        if low_mask.sum() > 0:
            print(f"    Before cal: P<0.1 actual positive rate = {y1v.values[low_mask].mean():.0%}")
            print(f"    After cal:  P<0.1 -> calibrated P mean = {pc[low_mask].mean():.2f}")
        print()

        # Stage 2: 回归 (正例, log空间)
        print("  -- Stage 2: Regressor --")
        pos_tr = df_train[df_train["is_positive"] == 1]
        pos_va = df_val[df_val["is_positive"] == 1]
        y2t = np.log1p(pos_tr[TARGET].values)
        y2v = np.log1p(pos_va[TARGET].values)
        p2 = {"objective":"regression_l1","metric":"mae",
              "learning_rate":0.05,"num_leaves":63,"min_child_samples":20,
              "subsample":0.8,"colsample_bytree":0.8,
              "reg_alpha":0.1,"reg_lambda":1.0,
              "n_jobs":-1,"seed":SEED,"verbose":-1}
        d2 = lgb.Dataset(pos_tr[features], label=y2t)
        d2v = lgb.Dataset(pos_va[features], label=y2v, reference=d2)
        self.reg_model = lgb.train(
            p2, d2, num_boost_round=5000, valid_sets=[d2v],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        print(f"    best_iter={self.reg_model.best_iteration}")

        # Lognormal 修正: E[exp(X)] = exp(μ + σ²/2) 而非 exp(μ)
        # log 空间残差方差越大 → 还原时低估越严重 → 需要更大修正
        lp_val = self.reg_model.predict(pos_va[features])
        self.log_sigma2 = float(np.var(y2v - lp_val))
        ln_factor = np.exp(min(self.log_sigma2 / 2, 0.5))  # 最大 e^0.5 ≈ 1.65x
        print(f"    log_sigma2={self.log_sigma2:.3f}  lognormal_factor={ln_factor:.3f}x")
        print()

    def _raw_predict(self, X):
        pr = self.cls_model.predict(X)
        pc = self.calibrator.predict(pr)
        lp = self.reg_model.predict(X)
        # 应用 lognormal 修正: 补偿 Jensen 不等式导致的系统低估
        sigma2 = getattr(self, "log_sigma2", 0.0)
        lp_corrected = lp + min(sigma2 / 2, 0.5)  # 与训练时相同的上限 e^0.5
        rp = np.clip(np.expm1(lp_corrected), 0, None)

        # V4: 栅栏 + 不乘P
        y = rp.copy()
        y[pc < self.threshold] = 0
        return y, pc, rp

    def optimize_threshold(self, df_val, features):
        print("  -- Threshold optimization --")
        X = df_val[features]
        yt = df_val[TARGET].values
        pr = self.cls_model.predict(X)
        pc = self.calibrator.predict(pr)
        lp = self.reg_model.predict(X)
        rp = np.clip(np.expm1(lp), 0, None)

        best_score, best_t = float("inf"), 0.30
        print(f"    {'Thr':>5s}  {'MAE':>6s}  {'Bias':>7s}  {'Zero%':>6s}  {'FN':>5s}  {'Score':>7s}")
        print(f"    {'-'*46}")

        for t in np.arange(0.15, 0.65, 0.05):
            yp = rp.copy()
            yp[pc < t] = 0
            yp = np.round(yp)
            mae  = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            zr   = (yp == 0).mean()
            fn   = ((yt > 0) & (yp == 0)).sum()
            # 综合得分: MAE + 漏报惩罚(负偏差代价是正偏差的2倍)
            # 这样优化器会倾向于选择偏差更接近0或略正的阈值，减少漏报
            score = mae + 0.5 * max(0.0, -bias)
            mk = " <" if score < best_score else ""
            if score < best_score:
                best_score, best_t = score, t
            print(f"    {t:5.2f}  {mae:6.3f}  {bias:+7.3f}  {zr:5.1%}  {fn:>5d}  {score:7.3f}{mk}")

        self.threshold = best_t
        print(f"    Best: {best_t:.2f} (score={best_score:.3f})")
        print()

    def learn_bias_correction(self, df_val, features):
        """
        分层偏差校正 — 基于 预测值 的层级 (推理时也能用)

        V7 修改:
          - clip 范围收窄: 0.5~2.0 → 0.8~1.3 (防止过度校正)
          - y>=6 区间只允许向上校正 (f>=1.0), 不允许向下压缩
            根因: val 集高需求被压缩后, test 集高需求也跟着低估
          - 最小样本量从 20 提高到 30
        """
        print("  -- Bias correction learning --")
        X  = df_val[features]
        yt = df_val[TARGET].values
        yr, _, _ = self._raw_predict(X)
        yr = np.round(yr)

        bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
        self.bias_factors = {}

        for lo, hi in bins:
            mask = (yr >= lo) & (yr <= hi)
            if mask.sum() < 30:
                self.bias_factors[(lo, hi)] = 1.0
                continue
            pm = yr[mask].mean()
            tm = yt[mask].mean()
            f  = tm / max(pm, 0.01) if pm > 0.1 else 1.0

            # 放宽上限: 验证集数据支持的修正就应该被用到
            # y=3-5 bin 里混入了大量 y_true=6-10 的低估样本，
            # 导致这个 bin 的 tm >> pm，修正系数需要超过 1.3x
            if lo >= 6:
                f = np.clip(f, 1.0, 2.2)   # 之前 1.7 不够用
            else:
                f = np.clip(f, 0.7, 1.8)   # 之前 0.8~1.3 太窄

            self.bias_factors[(lo, hi)] = f
            direction = "↑" if f > 1.0 else ("↓" if f < 1.0 else "=")
            print(f"    pred={lo}-{hi:>3d}: pred={pm:.2f} true={tm:.2f} f={f:.3f} {direction}")
        print()

    def predict(self, X):
        yr, pc, rp = self._raw_predict(X)
        yc = yr.copy()

        if self.bias_factors:
            for (lo, hi), f in self.bias_factors.items():
                mask = (yr >= lo) & (yr <= hi)
                if mask.any():
                    yc[mask] = yr[mask] * f

        yf = np.clip(np.round(yc), 0, None).astype(int)
        # yc 是偏差校正后的 float（未取整），供 gen_views 在应用 global_scale 前使用
        return yf, pc, rp, yc

    def feature_importance(self, features):
        cg = self.cls_model.feature_importance(importance_type="gain")
        rg = self.reg_model.feature_importance(importance_type="gain")
        fi = pd.DataFrame({"feature": features,
                           "cls": cg / max(cg.sum(),1) * 100,
                           "reg": rg / max(rg.sum(),1) * 100})
        fi["combined"] = fi["cls"] * 0.3 + fi["reg"] * 0.7
        return fi.sort_values("combined", ascending=False).reset_index(drop=True)


# ============================================================================
# Tweedie
# ============================================================================
class TweedieModel:
    def __init__(self):
        self.model = None

    def fit(self, df_train, df_val, features):
        print("  -- Tweedie --")
        p = {"objective":"tweedie","tweedie_variance_power":1.5,
             "metric":"mae","learning_rate":0.05,"num_leaves":63,
             "min_child_samples":20,"subsample":0.8,"colsample_bytree":0.8,
             "reg_alpha":0.1,"reg_lambda":1.0,"n_jobs":-1,"seed":SEED,"verbose":-1}
        d = lgb.Dataset(df_train[features], label=df_train[TARGET].values.astype(float))
        dv = lgb.Dataset(df_val[features], label=df_val[TARGET].values.astype(float), reference=d)
        self.model = lgb.train(
            p, d, num_boost_round=5000, valid_sets=[dv],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        print(f"    best_iter={self.model.best_iteration}")
        print()

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


# ============================================================================
# V4 集成
# ============================================================================
class V4Ensemble:
    def __init__(self):
        self.hurdle = CalibratedHurdleModel()
        self.tweedie = TweedieModel()
        self.weights = [0.65, 0.35]
        self.features = None

    def fit(self, df_train, df_val, features):
        self.features = features
        print("=" * 70)
        print("V4 Training")
        print("=" * 70)

        self.hurdle.fit(df_train, df_val, features)
        self.hurdle.optimize_threshold(df_val, features)
        self.hurdle.learn_bias_correction(df_val, features)
        self.tweedie.fit(df_train, df_val, features)
        self._opt_weights(df_val, features)

    def _opt_weights(self, df_val, features):
        print("  -- Ensemble weights --")
        X = df_val[features]
        yt = df_val[TARGET].values
        ph, _, _, _ = self.hurdle.predict(X)
        pt = self.tweedie.predict(X)

        best_score, best_w = float("inf"), 0.65
        for wh in np.arange(0.3, 0.9, 0.05):
            wt = 1.0 - wh
            yp  = np.clip(np.round(wh*ph + wt*pt), 0, None)
            mae  = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            # 与阈值优化相同的评分：低估代价是高估的2倍
            score = mae + 0.5 * max(0.0, -bias)
            if score < best_score:
                best_score, best_w = score, wh

        self.weights = [best_w, 1.0 - best_w]
        print(f"    Hurdle={best_w:.0%} Tweedie={1-best_w:.0%} Score={best_score:.3f}")

        for nm, pred in [("Hurdle+BiasCorr", ph), ("Tweedie", pt)]:
            yp = np.clip(np.round(pred.astype(float)), 0, None)
            mae = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            print(f"    {nm:20s}: MAE={mae:.3f} Bias={bias:+.3f}")
        print()

    def predict(self, X):
        ph, pc, rp, ph_float = self.hurdle.predict(X)
        pt = self.tweedie.predict(X)
        wh, wt = self.weights
        combined = wh * ph + wt * pt
        # combined_float 用偏差校正的 float 与 tweedie 混合，供 global_scale 前缩放
        combined_float = wh * ph_float + wt * pt
        yf = np.clip(np.round(combined), 0, None).astype(int)
        return yf, {"p_cal": pc, "pred_hurdle": ph,
                     "pred_tweedie": pt, "combined_raw": combined,
                     "combined_float": combined_float}

    def feature_importance(self, features):
        fi = self.hurdle.feature_importance(features)
        tg = self.tweedie.model.feature_importance(importance_type="gain")
        tg = tg / max(tg.sum(),1) * 100
        fi["tweedie"] = tg
        fi["combined"] = fi["cls"]*0.15 + fi["reg"]*0.35 + fi["tweedie"]*0.50
        return fi.sort_values("combined", ascending=False).reset_index(drop=True)


# ============================================================================
# 评估
# ============================================================================
def cmetrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    r = yp - yt; ae = np.abs(r); total = max(np.sum(yt), 1)
    return {"mae":np.mean(ae), "rmse":np.sqrt(np.mean(r**2)),
            "bias":np.mean(r), "wmape":np.sum(ae)/total,
            "hit0":np.mean(ae==0), "hit1":np.mean(ae<=1), "hit2":np.mean(ae<=2),
            "n":len(yt)}


def evaluate(model, df_test, features, v1p=None, v2p=None, v3p=None):
    print("=" * 70)
    print("V4 Evaluation")
    print("=" * 70)

    X = df_test[features]
    yt = df_test[TARGET].values
    yp, det = model.predict(X)
    m = cmetrics(yt, yp)

    print(f"\n  MAE={m['mae']:.3f}  Bias={m['bias']:+.3f}  "
          f"+/-1={m['hit1']:.1%}  +/-2={m['hit2']:.1%}")

    # 回归到均值
    print(f"\n  -- Regression to Mean --")
    for lo,hi in [(0,0),(1,2),(3,5),(6,10),(11,20),(21,999)]:
        mask = (yt>=lo)&(yt<=hi)
        if mask.sum()<5: continue
        tm = yt[mask].mean()
        pm = yp[mask].mean()
        ratio = pm / max(tm, 0.01)
        s = " OK" if 0.85<=ratio<=1.15 else " WARN" if 0.7<=ratio<=1.3 else " BAD"
        print(f"    y={lo}-{hi:>3d}: actual={tm:.1f} pred={pm:.1f} ratio={ratio:.2f} (ideal=1.00){s}")

    # FN
    fn = ((yt>0)&(yp==0)).sum()
    fn_m = yt[(yt>0)&(yp==0)].mean() if fn>0 else 0
    print(f"\n  FN: {fn} rows (avg actual={fn_m:.1f})")

    # 按量级
    print(f"\n  -- By Demand Level --")
    for lo,hi,lab in [(0,0,"y=0"),(1,2,"y=1-2"),(3,5,"y=3-5"),(6,10,"y=6-10"),(11,999,"y=11+")]:
        mask = (yt>=lo)&(yt<=hi)
        if mask.sum()<5: continue
        bm = cmetrics(yt[mask], yp[mask])
        print(f"    {lab:10s}  MAE={bm['mae']:.2f}  Bias={bm['bias']:+.2f}  +/-1={bm['hit1']:.0%}")

    # 按周
    print(f"\n  -- By Week --")
    edf = df_test[ID_COLS].copy()
    edf["y_true"] = yt; edf["y_pred"] = yp
    edf["error"] = yp-yt; edf["abs_error"] = np.abs(edf["error"])
    edf["p_calibrated"] = det["p_cal"]
    for wk in sorted(df_test["year_week"].unique()):
        s = edf[edf["year_week"]==wk]
        wm = cmetrics(s["y_true"].values, s["y_pred"].values)
        print(f"    {wk}: MAE={wm['mae']:.3f} Bias={wm['bias']:+.3f}")

    # 多版本对比
    print(f"\n  {'='*60}")
    print(f"  V1 / V2 / V3 / V4")
    print(f"  {'='*60}")
    v4wks = set(edf["year_week"].unique())
    comp = {"V4": m}
    for nm, pp in [("V1",v1p),("V2",v2p),("V3",v3p)]:
        if pp and Path(pp).exists():
            prev = pd.read_csv(pp)
            prev = prev[prev["year_week"].isin(v4wks)]
            if len(prev)>0:
                comp[nm] = cmetrics(prev["y_true"].values, prev["y_pred"].values)

    vers = [v for v in ["V1","V2","V3","V4"] if v in comp]
    print(f"\n  {'':12s}" + "".join(f"  {v:>8s}" for v in vers))
    print(f"  {'-'*12}" + "  --------" * len(vers))
    for metric, fmt in [("mae", ".3f"), ("bias", ".3f"), ("hit1", ".1%"), ("hit2", ".1%")]:
        row = f"  {metric:12s}"
        for v in vers:
            row += f"  {comp[v][metric]:>8{fmt}}"
        print(row)

    # 回归到均值对比
    print(f"\n  Regression-to-mean comparison:")
    for lo,hi in [(0,0),(1,2),(3,5),(6,10),(11,999)]:
        row = f"    y={lo}-{hi:>3d}:"
        for nm, pp in [("V1",v1p),("V2",v2p),("V3",v3p)]:
            if pp and Path(pp).exists():
                prev = pd.read_csv(pp)
                prev = prev[prev["year_week"].isin(v4wks)]
                pm = prev[(prev["y_true"]>=lo)&(prev["y_true"]<=hi)]
                if len(pm)>=5:
                    r = pm["y_pred"].mean() / max(pm["y_true"].mean(), 0.01)
                    row += f"  {nm}={r:.2f}"
        m4 = edf[(edf["y_true"]>=lo)&(edf["y_true"]<=hi)]
        if len(m4)>=5:
            r4 = m4["y_pred"].mean() / max(m4["y_true"].mean(), 0.01)
            row += f"  V4={r4:.2f}"
        print(row)
    print()
    return edf, m


# ============================================================================
# 样本外验证 (截断周预测 vs 真实总量)
# ============================================================================
def evaluate_holdout(model, df_holdout, features, actual_totals=None):
    """
    用训练好的模型预测截断周(样本外), 与真实总量对比.

    actual_totals: dict, e.g. {"2026-W09": 18250}
    """
    if df_holdout is None or df_holdout.empty:
        print("  (无截断周数据, 跳过样本外验证)")
        return df_holdout

    print("=" * 70)
    print("Holdout Prediction (Out-of-Sample Validation)")
    print("=" * 70)

    av = [c for c in features if c in df_holdout.columns]
    yp, _ = model.predict(df_holdout[av])
    df_out = df_holdout.copy()
    df_out["y_pred"] = yp

    for yw, gdf in df_out.groupby("year_week"):
        pred_total = gdf["y_pred"].sum()
        actual = (actual_totals or {}).get(str(yw))
        if actual:
            ratio = pred_total / actual
            diff  = pred_total - actual
            print(f"  {yw}: predicted={pred_total:>8,.0f}  actual={actual:>8,}  "
                  f"ratio={ratio:.2f}  diff={diff:+,.0f}")
        else:
            print(f"  {yw}: predicted={pred_total:>8,.0f}  (no actual provided)")

    out_path = OUTPUT_DIR / "holdout_predictions_v7.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}\n")
    return df_out


# ============================================================================
# 输出视图
# ============================================================================
def gen_views(model, df_full, features, global_scale: float = 1.0):
    """
    global_scale: 全局乘数，用于补偿模型系统性低估。
      推导方式: W09 holdout 实测 predicted=11211 vs actual=18250 → 18250/11211=1.627
      保守取 1.50 防止过校正，等积累更多实际数据后再更新。

    V7.2 修复: global_scale 现在作用于 float 预测值（取整前），
    而非整数预测值（取整后）。这样可以把许多原本被四舍五入为 0 的
    门店"救活"，显著增加司机视图中的门店数量。
    例如: combined_float=0.4, scale=1.5 → 0.60 → 取整为 1（之前是 0×1.5=0）
    """
    print("=" * 70)
    print("Generate Views")
    print("=" * 70)
    lw  = df_full["year_week"].max()
    lat = df_full[df_full["year_week"] == lw].copy()
    av  = [c for c in features if c in lat.columns]

    # 获取原始 float 预测值（combined_float = float hurdle + tweedie，未取整）
    _, det = model.predict(lat[av])
    yp_float = det.get("combined_float", det["combined_raw"])

    # 在 float 层面应用 global_scale，然后再取整 → 更多门店进入视图
    if global_scale != 1.0:
        yp_float = yp_float * global_scale
        print(f"  [global_scale={global_scale:.3f} applied to float predictions]")
    yp = np.clip(np.round(yp_float), 0, None).astype(int)

    lat["pw"] = yp
    lat["pl"] = np.clip(np.floor(yp * 0.85).astype(int), 0, None)
    lat["pu"] = np.ceil(yp * 1.15).astype(int) + 1
    print(f"  week={lw}  total={yp.sum():,.0f}  zeros={int((yp==0).sum())}")

    dr = {"Sunday":0.30,"Monday":0.10,"Tuesday":0.22,"Wednesday":0.08,"Thursday":0.30}
    rows = []
    for _, r in lat.iterrows():
        wp, wl, wu = r["pw"], r["pl"], r["pu"]
        rem, rl, ru = wp, wl, wu
        days = list(dr.keys())
        for i, (d, ratio) in enumerate(dr.items()):
            if i == len(days) - 1:
                dp, dl, dh = rem, rl, ru
            else:
                dp = int(round(wp * ratio))
                dl = int(round(wl * ratio))
                dh = int(round(wu * ratio))
                rem -= dp; rl -= dl; ru -= dh
            rows.append({
                "namn": r["namn"], "ort": r["ort"],
                "typ": r["typ"],   "sort": r["sort"],
                "product_id": r["product_id"],
                "year_week": r["year_week"], "day": d, "day_order": i,
                "predicted": max(0, dp),
                "pred_lower": max(0, dl),
                "pred_upper": max(0, dh),
            })
    daily = pd.DataFrame(rows)

    # ── 厨房视图（按产品汇总，不变）──────────────────────────────────
    do_map = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4}
    kit = (daily.groupby(["day","typ","sort","product_id"], as_index=False)
           .agg(total=("predicted","sum"), lower=("pred_lower","sum"),
                upper=("pred_upper","sum"), stores=("namn","nunique")))
    kit["do"] = kit["day"].map(do_map)
    kit = kit.sort_values(["do","total"], ascending=[True,False])
    kit["interval"] = kit["lower"].astype(str) + "~" + kit["upper"].astype(str)
    ko = kit[["day","typ","sort","total","interval","stores"]].rename(
        columns={"day":"Day","typ":"Type","sort":"Product",
                 "total":"Qty","interval":"Range","stores":"Stores"})

    # ── 司机视图（门店粒度）───────────────────────────────────────────
    # 只保留有预测量的行
    daily_pos = daily[daily["predicted"] > 0].copy()

    # 每个(路线, 门店, 工作日) 的产品明细字符串
    def build_detail(grp):
        parts = sorted([
            f"{row['typ']}/{row['sort']}: {int(row['predicted'])}"
            for _, row in grp.iterrows()
        ])
        return " | ".join(parts)

    detail_map = (
        daily_pos.groupby(["ort","namn","day"])
        .apply(build_detail, include_groups=False)
        .rename("product_detail")
        .reset_index()
    )

    # 门店汇总
    drv = (daily_pos.groupby(["ort","namn","day","day_order"], as_index=False)
           .agg(total_items  =("predicted",    "sum"),
                lower_total  =("pred_lower",   "sum"),
                upper_total  =("pred_upper",   "sum"),
                n_products   =("product_id",   "nunique")))
    drv = drv.merge(detail_map, on=["ort","namn","day"], how="left")
    drv["interval"] = drv["lower_total"].astype(str) + "~" + drv["upper_total"].astype(str)
    drv = drv.sort_values(["ort","day_order","namn"])
    dvo = drv[["ort","namn","day","total_items","interval","n_products","product_detail"]].rename(
        columns={"ort":"Route","namn":"Store","day":"Day",
                 "total_items":"Total_Qty","interval":"Range",
                 "n_products":"N_Products","product_detail":"Products"})

    print(f"  厨房视图: {len(ko)} 行 (工作日×产品)")
    print(f"  司机视图: {len(dvo)} 行 (路线×门店×工作日)  ← 原来几千行\n")
    return ko, dvo


# ============================================================================
# 诊断图
# ============================================================================
def plot_v4(edf, fi, v1p=None, v2p=None, v3p=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed"); return

    print("=" * 70)
    print("V4 Diagnostics Plot")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Eataway V7 — Calibrated Hurdle + Bias Correction + Driver View", fontsize=14, y=0.98)

    # 1. Pred vs Actual
    ax = axes[0,0]
    s = edf.sample(min(5000,len(edf)), random_state=SEED)
    ax.scatter(s["y_true"], s["y_pred"], alpha=0.15, s=10, c="#2196F3")
    mx = max(s["y_true"].max(), s["y_pred"].max())
    ax.plot([0,mx],[0,mx],"r--",lw=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual (V4)")

    # 2. Error dist
    ax = axes[0,1]
    ax.hist(edf["error"], bins=np.arange(-10.5,11.5,1), color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Error"); ax.set_title(f"Error Distribution (mean={edf['error'].mean():.2f})")
    ax.set_xlim(-10,10)

    # 3. Regression-to-mean comparison
    ax = axes[0,2]
    labels = ["0","1-2","3-5","6-10","11+"]
    cuts = [-0.5,0.5,2.5,5.5,10.5,200]
    v4wks = set(edf["year_week"].unique())
    x = np.arange(len(labels)); width=0.18
    colors = {"V1":"#FF5722","V2":"#2196F3","V3":"#FF9800","V4":"#9C27B0"}

    for i,(nm,pp) in enumerate([("V1",v1p),("V2",v2p),("V3",v3p)]):
        if pp and Path(pp).exists():
            prev = pd.read_csv(pp)
            prev = prev[prev["year_week"].isin(v4wks)]
            if len(prev)>0:
                prev["db"] = pd.cut(prev["y_true"], bins=cuts, labels=labels)
                ratios = []
                for lb in labels:
                    sb = prev[prev["db"]==lb]
                    ratios.append(min(sb["y_pred"].mean()/max(sb["y_true"].mean(),0.01), 3.0) if len(sb)>=5 else np.nan)
                ax.bar(x+i*width, ratios, width, label=nm, color=colors[nm], alpha=0.6)

    edf["db"] = pd.cut(edf["y_true"], bins=cuts, labels=labels)
    r4 = []
    for lb in labels:
        sb = edf[edf["db"]==lb]
        r4.append(min(sb["y_pred"].mean()/max(sb["y_true"].mean(),0.01), 3.0) if len(sb)>=5 else np.nan)
    ax.bar(x+3*width, r4, width, label="V4", color=colors["V4"], alpha=0.9)
    ax.axhline(1.0,color="red",ls="--",lw=1)
    ax.set_xticks(x+1.5*width); ax.set_xticklabels(labels)
    ax.set_ylabel("Pred/Actual Ratio"); ax.set_title("Regression-to-Mean: V1->V4")
    ax.legend(fontsize=8)

    # 4. Feature importance
    ax = axes[1,0]
    top = fi.head(15).iloc[::-1]
    ax.barh(top["feature"], top["combined"], color="#FF9800", edgecolor="white")
    ax.set_xlabel("Importance (%)"); ax.set_title("Top 15 Features (V4)")

    # 5. MAE by level
    ax = axes[1,1]
    mae4 = edf.groupby("db", observed=True)["abs_error"].mean()
    ax.bar(x, mae4.reindex(labels).values, color="#9C27B0", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Demand Level"); ax.set_ylabel("MAE"); ax.set_title("MAE by Level (V4)")

    # 6. Bias by week
    ax = axes[1,2]
    wb = edf.groupby("year_week")["error"].mean()
    ax.bar(range(len(wb)), wb.values, color="#00BCD4", edgecolor="white", alpha=0.8)
    ax.axhline(0,color="red",ls="--")
    ax.set_xticks(range(len(wb))); ax.set_xticklabels(wb.index, rotation=45, fontsize=8)
    ax.set_ylabel("Bias"); ax.set_title("Bias by Week (V4)")

    plt.tight_layout()
    p = OUTPUT_DIR / "diagnostics_v7.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}\n")


# ============================================================================
# 保存
# ============================================================================
def save_all(model, edf, metrics, fi, kit, drv):
    print("=" * 70)
    print("Save")
    print("=" * 70)

    model.hurdle.cls_model.save_model(str(OUTPUT_DIR / "model_cls.txt"))
    model.hurdle.reg_model.save_model(str(OUTPUT_DIR / "model_reg.txt"))
    model.tweedie.model.save_model(str(OUTPUT_DIR / "model_tweedie.txt"))

    import pickle
    with open(OUTPUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(model.hurdle.calibrator, f)

    edf.to_csv(OUTPUT_DIR / "evaluation_v7.csv", index=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance_v7.csv", index=False)
    kit.to_csv(OUTPUT_DIR / "kitchen_view_v7.csv", index=False, encoding="utf-8-sig")
    drv.to_csv(OUTPUT_DIR / "driver_view_v7.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics_v7.csv", index=False)

    config = {
        "hurdle_threshold": model.hurdle.threshold,
        "bias_factors": {f"{lo}-{hi}": float(f) for (lo,hi),f in model.hurdle.bias_factors.items()},
        "ensemble_weights": {"hurdle": model.weights[0], "tweedie": model.weights[1]},
        "n_features": len(model.features) if model.features else 0,
        "has_weather": any("temp" in f for f in (model.features or [])),
    }
    with open(OUTPUT_DIR / "config_v4.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("  All saved\n")


# ============================================================================
# main
# ============================================================================
def main():
    print("=" * 70)
    print("  EATAWAY V7")
    print("  Calibrated Hurdle + Fixed Bias Correction + Store-Level Driver View")
    print("=" * 70 + "\n")

    df, features, df_holdout = load_and_prepare()
    dtr, dva, dte = time_split(df)

    model = V4Ensemble()
    model.fit(dtr, dva, features)

    v1p = None   # 旧电脑路径，跳过版本对比
    v2p = None
    v3p = None

    edf, metrics = evaluate(model, dte, features, v1p, v2p, v3p)

    fi = model.feature_importance(features)
    print("=" * 70)
    print("Feature Importance Top 20")
    print("=" * 70)
    for i, row in fi.head(20).iterrows():
        bar = "#" * max(1, int(row["combined"]))
        print(f"  {i+1:>2d}. {row['feature']:30s}  {row['combined']:5.1f}%  {bar}")
    print()

    # ── 样本外验证 ──────────────────────────────────────────────────
    KNOWN_ACTUALS = {"2026-W09": 18250}
    evaluate_holdout(model, df_holdout, features, KNOWN_ACTUALS)

    # ── 用最后一个完整训练周做"干净"预测 ─────────────────────────────
    # 截断周(W08/W09/W10)的 lag 特征被前一截断周的不完整数据污染:
    #   W09 lag_1w = W08_truncated(11,412) 而非 W08_actual(~14,000+)
    # 所以直接用 holdout 行预测是不可靠的。
    # 改用训练集最后一个完整周(W07)的行: 特征100%干净(lag=W06完整数据)
    last_clean_week = df["year_week"].max()
    clean_rows = df[df["year_week"] == last_clean_week]
    av = [c for c in features if c in clean_rows.columns]
    yp_clean, _ = model.predict(clean_rows[av])
    clean_total = yp_clean.sum()
    print("=" * 70)
    print("Clean Proxy Prediction (last complete training week)")
    print("=" * 70)
    print(f"  base_week={last_clean_week}  predicted_total={clean_total:,.0f}")
    print(f"  (W09 actual=18,250  proxy_ratio={clean_total/18250:.2f})")
    print()

    # ── 全局缩放系数 ───────────────────────────────────────────────
    # 测试集自动系数 + 春季需求补偿
    total_actual    = edf["y_true"].sum()
    total_predicted = edf["y_pred"].sum()
    auto_scale = total_actual / max(total_predicted, 1.0)
    # 下限 1.30: 弥补模型对高需求期（春季等）的系统性低估
    # 上限 1.60: 防止过校正
    # 随着积累更多年份数据，模型学到季节性后可以把下限降回 1.0
    VIEW_GLOBAL_SCALE = float(np.clip(auto_scale, 1.40, 1.70))
    print(f"  Auto global_scale = {total_actual:.0f} / {total_predicted:.0f} = {auto_scale:.3f} → clipped={VIEW_GLOBAL_SCALE:.3f}\n")

    # ── 生成输出视图 (用完整训练周，特征干净) ────────────────────────
    kit, drv = gen_views(model, df, features, global_scale=VIEW_GLOBAL_SCALE)
    plot_v4(edf, fi, v1p, v2p, v3p)
    save_all(model, edf, metrics, fi, kit, drv)

    print("=" * 70)
    print("  V4 DONE")
    print(f"  Output: {OUTPUT_DIR}")
    wh, wt = model.weights
    print(f"  Hurdle {wh:.0%} + Tweedie {wt:.0%}")
    print(f"  Threshold: {model.hurdle.threshold:.2f}")
    hw = any("temp" in f for f in features)
    print(f"  Weather: {'YES' if hw else 'NO (run eataway_weather.py first)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()