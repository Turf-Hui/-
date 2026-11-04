# -*- coding: utf-8 -*-
"""
03_analysis.py —— 商业分析模型（核心模块）
========================================
应用 ≥3 种商业分析模型（本项目实际应用 8 种），每种产出核心结果表：
  1) RFM 八分类            5) 关联规则 Apriori
  2) 同期群留存热力图数据    6) Holt-Winters 时间序列预测（含置信区间 + 回测误差）
  3) 帕累托/ABC 分类        7) K-Means 客户分群
  4) 波士顿矩阵(BCG)        8) 中文评论情感分析
模型参数/阈值均给出业务依据（见论文第 4 章与各函数注释）。
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(__file__), "..")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
TABLE_DIR = os.path.join(BASE, "outputs", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)


def _save(df, name):
    path = os.path.join(TABLE_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ----------------------------- 1. RFM 八分类 -----------------------------
def rfm_analysis(fact: pd.DataFrame):
    fact = fact[fact["is_return"] == 0].copy()
    fact["order_date"] = pd.to_datetime(fact["order_date"])
    snapshot = fact["order_date"].max() + pd.Timedelta(days=1)

    rfm = fact.groupby("customer_id").agg(
        Recency=("order_date", lambda s: (snapshot - s.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("net_amount", "sum"),
    ).reset_index()

    # 阈值依据：用中位数切分高低（业务上"高于中位数=高价值/活跃"）。
    # 真实零售数据中 Frequency 中位数常为 1（多数客户仅购买一次），若用 >= 会使
    # 全部客户被判"高频"、八分类退化为四类；故 F 采用「严格大于中位数」，
    # 即 F>=2 视为复购客户，业务含义更清晰，也能完整得到 8 个细分类。
    r_med, f_med, m_med = rfm["Recency"].median(), rfm["Frequency"].median(), rfm["Monetary"].median()
    rfm["R"] = np.where(rfm["Recency"] <= r_med, "高", "低")   # 最近购买=高
    rfm["F"] = np.where(rfm["Frequency"] > f_med, "高", "低")  # 严格大于：复购客户=高
    rfm["M"] = np.where(rfm["Monetary"] >= m_med, "高", "低")

    seg_map = {
        ("高", "高", "高"): "重要价值客户", ("高", "高", "低"): "重要发展客户",
        ("高", "低", "高"): "重要保持客户", ("高", "低", "低"): "一般新客户",
        ("低", "高", "高"): "重要挽留客户", ("低", "高", "低"): "一般潜力客户",
        ("低", "低", "高"): "一般维持客户", ("低", "低", "低"): "流失风险客户",
    }
    rfm["segment"] = rfm.apply(lambda r: seg_map[(r["R"], r["F"], r["M"])], axis=1)
    # 业务评分（1-5 分位）用于 K-Means 与可视化
    # 真实交易数据的 Recency 存在大量并列值，直接 qcut 会因分箱边界重复报错，
    # 故与 F/M 一致先按 rank(method="first") 打散并列，再等频分 5 箱。
    rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm.to_csv(os.path.join(CLEAN_DIR, "rfm.csv"), index=False, encoding="utf-8-sig")
    seg = rfm["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    seg.to_csv(os.path.join(TABLE_DIR, "rfm_segments.csv"), index=False, encoding="utf-8-sig")
    return rfm, seg


# ----------------------------- 2. 同期群留存 -----------------------------
def cohort_analysis(fact: pd.DataFrame, customers: pd.DataFrame):
    cust = customers.copy()
    cust["register_month"] = pd.to_datetime(cust["register_date"]).dt.to_period("M").astype(str)
    ord_ = fact[fact["is_return"] == 0][["customer_id", "order_date"]].copy()
    ord_["order_month"] = pd.to_datetime(ord_["order_date"]).dt.to_period("M").astype(str)
    df = ord_.merge(cust[["customer_id", "register_month"]], on="customer_id", how="left")
    df = df.dropna(subset=["register_month"])
    pi1 = pd.PeriodIndex(df["order_month"], freq="M").astype("int64")
    pi2 = pd.PeriodIndex(df["register_month"], freq="M").astype("int64")
    df["cohort_index"] = (pi1 - pi2).astype(int)
    # 只取首单月（新客激活）作为留存基准
    first = df.groupby(["customer_id", "register_month"])["cohort_index"].min().reset_index()
    first = first[first["cohort_index"] == 0][["customer_id", "register_month"]]
    active = df[df["cohort_index"] >= 0].groupby(["customer_id", "register_month", "cohort_index"]).size().reset_index(name="x")
    active = active.merge(first, on=["customer_id", "register_month"], how="inner")
    cohort = active.pivot_table(index="register_month", columns="cohort_index", values="customer_id", aggfunc="nunique")
    cohort = cohort.sort_index()
    # 留存率 = 各期活跃数 / 该同期群基数(第0期)
    cohort_size = cohort[0]
    cohort_pct = cohort.div(cohort_size, axis=0)
    cohort_pct = cohort_pct.round(3)
    cohort_pct.to_csv(os.path.join(TABLE_DIR, "cohort_retention.csv"))
    return cohort_pct


# ----------------------------- 3. 帕累托 / ABC -----------------------------
def pareto_analysis(fact: pd.DataFrame):
    rev = fact[fact["is_return"] == 0].groupby("product_category")["net_amount"].sum().sort_values(ascending=False)
    cum = rev.cumsum() / rev.sum()
    abc = pd.DataFrame({"revenue": rev, "cum_pct": cum})
    abc["ABC"] = np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C"))
    abc = abc.reset_index().rename(columns={"product_category": "category"})
    _save(abc, "pareto_abc.csv")
    return abc


# ----------------------------- 4. 波士顿矩阵 BCG -----------------------------
def bcg_analysis(fact: pd.DataFrame):
    fact = fact.copy()
    fact["order_date"] = pd.to_datetime(fact["order_date"])
    rev = fact[fact["is_return"] == 0].groupby("product_category")["net_amount"].sum()
    # 增长率：近 6 个月均值 vs 前 6 个月均值（基于最近 12 个月的类目月度销售额）
    monthly_cat = (fact[fact["is_return"] == 0]
                   .pivot_table(index="order_month", columns="product_category",
                                values="net_amount", aggfunc="sum").fillna(0))
    last12 = monthly_cat.tail(12)
    recent = last12.tail(6).mean()
    prior = last12.head(6).mean()
    growth = (recent - prior) / prior.replace(0, np.nan)
    growth = growth.fillna(0)
    bcg = pd.DataFrame({"revenue": rev})
    bcg["growth"] = growth.reindex(bcg.index).fillna(0)
    max_rev = bcg["revenue"].max()
    bcg["rel_share"] = bcg["revenue"] / max_rev           # 相对市场份额（0-1）
    # 划分象限
    g_med, s_med = bcg["growth"].median(), bcg["rel_share"].median()
    def quad(r):
        if r["growth"] >= g_med and r["rel_share"] >= s_med: return "明星类(Star)"
        if r["growth"] < g_med and r["rel_share"] >= s_med: return "现金牛(Cash Cow)"
        if r["growth"] >= g_med and r["rel_share"] < s_med: return "问题类(Question)"
        return "瘦狗类(Dog)"
    bcg["quadrant"] = bcg.apply(quad, axis=1)
    bcg = bcg.reset_index().rename(columns={"product_category": "category"})
    _save(bcg, "bcg_matrix.csv")
    return bcg


# ----------------------------- 5. 关联规则 Apriori -----------------------------
def association_rules(fact: pd.DataFrame):
    """购物篮关联规则。

    UCI Online Retail 为批发型数据，单张发票常含数十个不同商品，若直接以「类目」
    构建购物篮，多数发票会覆盖几乎所有类目，导致 lift 趋近 1、无法产出有效强规则。
    故采用两级策略：
      ① 优先以「类目」购物篮（业务可解释性最强）；
      ② 若类目级无 lift>1 的规则，自动降级到「高频商品(StockCode)」购物篮，
         以真实商品共现挖掘搭配关系，并在结果表 rule_level 列注明所用层级。
    """
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules as _ar
    except Exception as e:
        print("[03] mlxtend 不可用，跳过关联规则:", e)
        return None

    def mine(baskets, min_support, min_lift, level):
        if not baskets:
            return None
        te = TransactionEncoder()
        arr = te.fit(baskets).transform(baskets)
        df = pd.DataFrame(arr, columns=te.columns_)
        try:
            freq = apriori(df, min_support=min_support, use_colnames=True, max_len=3)
        except Exception as e:
            print(f"[03] apriori({level}) 失败:", e)
            return None
        if freq.empty:
            return None
        rules = _ar(freq, metric="lift", min_threshold=min_lift)
        if rules.empty:
            return None
        rules = rules.sort_values("lift", ascending=False).head(30)
        out = rules[["antecedents", "consequents", "support", "confidence", "lift", "leverage"]].copy()
        out["antecedents"] = out["antecedents"].apply(lambda x: ",".join(sorted(map(str, x))))
        out["consequents"] = out["consequents"].apply(lambda x: ",".join(sorted(map(str, x))))
        out["rule_level"] = level
        return out

    # ① 类目级
    baskets_cat = fact.groupby("order_id")["product_category"].apply(lambda s: sorted(set(s))).tolist()
    out = mine(baskets_cat, min_support=0.01, min_lift=1.1, level="类目级")

    # ② 商品级兜底：只保留出现频次 Top 120 的真实商品，缓解稀疏性
    if out is None or out.empty:
        print("[03] 类目级无有效强规则，降级到高频商品级购物篮 ...")
        top = fact["product_id"].value_counts().head(120).index
        sub = fact[fact["product_id"].isin(top)]
        baskets_prd = sub.groupby("order_id")["product_id"].apply(lambda s: sorted(set(s))).tolist()
        out = mine(baskets_prd, min_support=0.008, min_lift=1.2, level="商品级(高频Top120)")

    if out is None or out.empty:
        print("[03] 无满足阈值的关联规则")
        return None
    _save(out, "association_rules.csv")
    print(f"[03] 关联规则 {len(out)} 条，层级={out['rule_level'].iloc[0]}，"
          f"最高 lift={out['lift'].max():.3f}")
    return out


# ----------------------------- 6. 时间序列预测（Holt-Winters）+ 回测 -----------------------------
def holt_winters_forecast(fact: pd.DataFrame):
    """销售预测：Holt-Winters 指数平滑（加法趋势 + 加法季节）。

    参数业务依据（重要，适配 UCI 数据特征）：
      UCI Online Retail 仅覆盖 2010-12-01~2011-12-09 约 12 个月，月度序列只有 13 个点，
      不足以估计 12 期年季节项（需 ≥2 个完整周期）。故按「周」聚合为约 53 个点，
      采用 seasonal_periods=4（近似「月内 4 周」的补货/结算节律），
      以末 8 周做样本外回测，输出点预测 + 95% 置信区间 + MAPE/RMSE。
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception as e:
        print("[03] statsmodels 不可用，跳过预测:", e)
        return None

    d = fact[fact["is_return"] == 0][["order_date", "net_amount"]].copy()
    d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
    d = d.dropna(subset=["order_date"])
    # resample 生成等间隔周序列；缺周填 0，避免布尔过滤破坏 DatetimeIndex.freq
    ts = (d.set_index("order_date").sort_index()
          .resample("W-MON")["net_amount"].sum().astype(float))
    ts = ts.replace(0.0, np.nan).interpolate().fillna(0.0)

    SEASON, H = 4, 8
    if len(ts) < 3 * SEASON:
        print("[03] 周度序列过短，跳过预测")
        return None
    train, test = ts.iloc[:-H], ts.iloc[-H:]

    def fit(s):
        # 传入 numpy 数组而非 Series，规避 statsmodels 对 index 频率的严格要求
        return ExponentialSmoothing(np.asarray(s, dtype=float), trend="add", seasonal="add",
                                    seasonal_periods=SEASON,
                                    initialization_method="estimated").fit(optimized=True)

    try:
        model = fit(train)
        fc = np.asarray(model.forecast(H), dtype=float)
        mape = float(np.mean(np.abs((test.values - fc) / test.values)) * 100)
        rmse = float(np.sqrt(np.mean((test.values - fc) ** 2)))
        resid = np.asarray(model.resid, dtype=float)
        resid_std = float(np.nanstd(resid))
        if not np.isfinite(resid_std) or resid_std == 0:
            resid_std = float(np.std(train.values))
        fut = np.asarray(fit(ts).forecast(H), dtype=float)
        steps = np.arange(1, H + 1)
        band = 1.96 * resid_std * np.sqrt(steps)   # 95% 置信区间随步长放大
        step = pd.Timedelta(weeks=1)
        idx_fut = [ts.index[-1] + step * i for i in range(1, H + 1)]
        labels = [x.strftime("%Y-%m-%d") for x in list(test.index)] + \
                 [x.strftime("%Y-%m-%d") for x in idx_fut]
        result = pd.DataFrame({
            "month": labels,                       # 兼容 04_visuals 的 x 轴字段名（值为周起始日）
            "period": labels,                      # 语义化列名：周度
            "type": ["actual"] * H + ["forecast"] * H,
            "value": list(test.values) + list(fut),
            "lower": [np.nan] * H + list((fut - band)),
            "upper": [np.nan] * H + list((fut + band)),
        })
        _save(result, "holt_winters_forecast.csv")
        meta = {"MAPE": round(mape, 2), "RMSE": round(rmse, 2),
                "backtest_period": "末8周", "granularity": "周度(W-MON)",
                "model": "Holt-Winters(add,add,4)",
                "n_obs": int(len(ts)),
                "note": "UCI 数据约12个月，月度序列不足以拟合12期季节项，故改用周度+4周季节"}
        pd.DataFrame([meta]).to_csv(os.path.join(TABLE_DIR, "holt_winters_metric.csv"), index=False)
        return result, meta
    except Exception as e:
        print("[03] Holt-Winters 拟合失败:", e)
        return None


# ----------------------------- 7. K-Means 客户分群 -----------------------------
def kmeans_segment(rfm: pd.DataFrame):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except Exception as e:
        print("[03] sklearn 不可用，跳过 K-Means:", e)
        return None
    X = rfm[["R_score", "F_score", "M_score"]].values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xs)
    rfm["cluster"] = km.labels_
    prof = rfm.groupby("cluster").agg(
        n=("customer_id", "size"),
        R=("Recency", "mean"), F=("Frequency", "mean"), M=("Monetary", "mean"),
    ).round(2).reset_index()
    _save(prof, "kmeans_profile.csv")
    rfm.to_csv(os.path.join(CLEAN_DIR, "rfm.csv"), index=False, encoding="utf-8-sig")
    return prof


# ----------------------------- 8. 中文评论情感分析（本地词表法，无需联网模型） -----------------------------
POS_WORDS = ["很快", "完好", "满意", "超出", "预期", "性价比", "高", "推荐", "及时", "棒", "不错",
             "精美", "扎实", "物超所值", "喜欢", "质量", "好", "体验", "完美", "赞"]
NEG_WORDS = ["慢", "破损", "堪忧", "不符", "失望", "差", "坏", "不推荐", "低", "一般", "问题",
             "退货", "敷衍", "贵", "坑", "垃圾", "难受"]


def sentiment_analysis(reviews: pd.DataFrame):
    """本地中文情感词表打分：正面词得分 +1、负面词得分 -1，归一化到 [0,1]。
    不依赖任何联网模型（snownlp/jieba），保证离线可复现。"""
    # 关联商品维度表以补充 product_category（原始评论表仅有 product_id）
    # UCI 的 StockCode 为纯数字，评论表读出为 int64 而维度表为 str，需统一类型避免 merge 报错
    products = pd.read_csv(os.path.join(CLEAN_DIR, "dim_products.csv"), dtype={"product_id": str})
    reviews = reviews.copy()
    reviews["product_id"] = reviews["product_id"].astype(str).str.strip()
    reviews = reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
    def score(text):
        t = str(text)
        s = sum(1 for w in POS_WORDS if w in t) - sum(1 for w in NEG_WORDS if w in t)
        denom = sum(1 for w in POS_WORDS if w in t) + sum(1 for w in NEG_WORDS if w in t)
        if denom == 0:
            return 0.5
        return round((s + denom) / (2 * denom), 3)
    reviews = reviews.copy()
    reviews["sentiment"] = reviews["comment_text"].apply(score)
    reviews["label"] = np.where(reviews["sentiment"] >= 0.6, "正面",
                        np.where(reviews["sentiment"] <= 0.4, "负面", "中性"))
    summary = reviews.groupby("label").size().rename_axis("label").reset_index(name="count")
    by_cat = reviews.groupby("category")["sentiment"].mean().round(3).reset_index().rename(columns={"sentiment": "avg_sentiment"})
    _save(summary, "sentiment_summary.csv")
    _save(by_cat, "sentiment_by_category.csv")
    reviews.to_csv(os.path.join(CLEAN_DIR, "fact_reviews_sentiment.csv"), index=False, encoding="utf-8-sig")
    return summary, by_cat


# ----------------------------- 主流程 -----------------------------
def main():
    print("[03_analysis] 读取清洗后数据 ...")
    fact = pd.read_csv(os.path.join(CLEAN_DIR, "fact_orders.csv"))
    customers = pd.read_csv(os.path.join(CLEAN_DIR, "dim_customers.csv"))
    reviews = pd.read_csv(os.path.join(CLEAN_DIR, "fact_reviews.csv"))

    print("  · RFM 八分类 ...")
    rfm, seg = rfm_analysis(fact)
    print(seg.to_string(index=False))

    print("  · 同期群留存 ...")
    cohort = cohort_analysis(fact, customers)
    cohort.to_csv(os.path.join(TABLE_DIR, "cohort_retention.csv"))

    print("  · 帕累托/ABC ...")
    pareto_analysis(fact)

    print("  · 波士顿矩阵 ...")
    bcg_analysis(fact)

    print("  · 关联规则 Apriori ...")
    association_rules(fact)

    print("  · Holt-Winters 预测 ...")
    holt_winters_forecast(fact)

    print("  · K-Means 分群 ...")
    kmeans_segment(rfm)

    print("  · 中文评论情感分析 ...")
    sentiment_analysis(reviews)

    print("[03_analysis] 全部模型结果已写入:", os.path.abspath(TABLE_DIR))


if __name__ == "__main__":
    main()
