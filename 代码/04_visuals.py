# -*- coding: utf-8 -*-
"""
04_visuals.py —— 数据可视化（全 Python 实现）
============================================
产出要求：
  - 图表总数 ≥10 张；模板复现 ≥10 种；业务模型图 ≥6 张（与 03 模型一一对应）
  - 分辨率 ≥150 dpi，统一命名 fig01_xxx.png，中文字体正常显示
对照《Excel数据可视化——从图表到数据大屏》典型模板，使用 Matplotlib/Seaborn 复现。
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# -------------------- 中文字体（评分要求：中文字体正常显示） --------------------
# 优先使用「微软雅黑」（含独立粗体 msyhbd.ttc），避免 SimHei 无粗体导致的警告
for _fp in [r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/msyhbd.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
YAHEI = "Microsoft YaHei"
if os.path.exists(r"C:/Windows/Fonts/msyh.ttc"):
    plt.rcParams["font.family"] = YAHEI
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

BASE = os.path.join(os.path.dirname(__file__), "..")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
TABLE_DIR = os.path.join(BASE, "outputs", "tables")
FIG_DIR = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 统一配色（评分要求：配色协调统一）
PALETTE = ["#2E5E8C", "#E07B39", "#3C896D", "#C1392B", "#8E44AD", "#16A085", "#D4AC0D", "#5D6D7E"]
ACCENT = "#2E5E8C"


def _yfmt(x, _):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def safe_read(name):
    """安全读取可选结果表：文件不存在（对应模型因缺库未运行）时返回 None。"""
    p = os.path.join(TABLE_DIR, name)
    if not os.path.exists(p):
        print(f"[04] 跳过（缺失 {name}，可能对应模型未运行）")
        return None
    return pd.read_csv(p)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def main():
    fact = pd.read_csv(os.path.join(CLEAN_DIR, "fact_orders.csv"), parse_dates=["order_date"])
    reviews = pd.read_csv(os.path.join(CLEAN_DIR, "fact_reviews_sentiment.csv"), parse_dates=["review_date"]) \
        if os.path.exists(os.path.join(CLEAN_DIR, "fact_reviews_sentiment.csv")) else pd.read_csv(os.path.join(CLEAN_DIR, "fact_reviews.csv"))
    rfm = pd.read_csv(os.path.join(CLEAN_DIR, "rfm.csv"))
    monthly = fact[fact["is_return"] == 0].groupby("order_month")["net_amount"].sum()

    # ---------- 模板复现 1：销售趋势折线图 ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o", color=ACCENT, lw=2)
    ax.set_title("图1 月度净销售额趋势（折线图）", fontsize=13, weight="bold")
    ax.set_ylabel("净销售额(英镑)"); ax.yaxis.set_major_formatter(FuncFormatter(_yfmt))
    ax.grid(alpha=.3); ax.tick_params(axis="x", rotation=45)
    save(fig, "fig01_销售趋势_折线图.png")

    # ---------- 模板复现 2：类目结构环形图 ----------
    cat = fact[fact["is_return"] == 0].groupby("product_category")["net_amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%", colors=PALETTE, pctdistance=.8, wedgeprops=dict(width=.42))
    ax.set_title("图2 类目销售占比（环形图）", fontsize=13, weight="bold")
    save(fig, "fig02_类目占比_环形图.png")

    # ---------- 模板复现 3：国家/地区分布柱状图（UCI 数据集的 Country 字段） ----------
    prov = fact[fact["is_return"] == 0].groupby("province")["net_amount"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(prov.index, prov.values, color=ACCENT)
    ax.set_title("图3 各国家/地区销售分布 Top15（柱状图）", fontsize=13, weight="bold")
    ax.set_ylabel("净销售额(英镑)"); ax.yaxis.set_major_formatter(FuncFormatter(_yfmt))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save(fig, "fig03_地区分布_柱状图.png")

    # ---------- 模板复现 4：渠道占比百分比堆叠 ----------
    ch = fact[fact["is_return"] == 0].groupby(["order_month", "channel"])["net_amount"].sum().unstack().fillna(0)
    chp = ch.div(ch.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(chp))
    for i, c in enumerate(chp.columns):
        ax.bar(chp.index, chp[c].values, bottom=bottom, label=c, color=PALETTE[i % len(PALETTE)])
        bottom += chp[c].values
    ax.set_title("图4 渠道销售占比（百分比堆叠柱）", fontsize=13, weight="bold")
    ax.legend(fontsize=8, ncol=4); ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    save(fig, "fig04_渠道占比_堆叠柱.png")

    # ---------- 模板复现 5：会员等级条形图 ----------
    ml = fact.groupby("member_level")["net_amount"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"L{i}" for i in ml.index], ml.values, color=PALETTE[2])
    ax.set_title("图5 会员等级销售贡献（条形图）", fontsize=13, weight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(_yfmt))
    save(fig, "fig05_会员等级_条形图.png")

    # ---------- 模板复现 6：价格分布直方图 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(fact["unit_price"], bins=40, color=PALETTE[3], alpha=.85)
    ax.set_title("图6 商品单价分布（直方图）", fontsize=13, weight="bold")
    ax.set_xlabel("单价(英镑)"); ax.set_ylabel("频数")
    save(fig, "fig06_价格分布_直方图.png")

    # ---------- 模板复现 7：客单价箱线图 ----------
    order_amt = fact[fact["is_return"] == 0].groupby("order_id")["net_amount"].sum()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(order_amt.values, vert=True, patch_artist=True,
               boxprops=dict(facecolor=PALETTE[1]), medianprops=dict(color="white"))
    ax.set_title("图7 订单客单价分布（箱线图）", fontsize=13, weight="bold")
    ax.set_ylabel("客单价(英镑)"); ax.set_xticks([])
    save(fig, "fig07_客单价_箱线图.png")

    # ---------- 模板复现 8：月度×类目热力图（矩阵） ----------
    pivot = fact[fact["is_return"] == 0].pivot_table(index="product_category", columns="order_month",
                                                     values="net_amount", aggfunc="sum").fillna(0)
    pivot = pivot.loc[:, pivot.columns[pivot.columns.str.endswith(("01","07"))]]  # 抽样月份便于展示
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("图8 类目×月份销售热力（矩阵热力图）", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    save(fig, "fig08_类目月份_热力图.png")

    # ---------- 业务图 9：帕累托双轴组合图 ----------
    abc = pd.read_csv(os.path.join(TABLE_DIR, "pareto_abc.csv"))
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(abc["category"], abc["revenue"], color=PALETTE[0])
    ax1.set_ylabel("销售额(英镑)", color=PALETTE[0]); ax1.yaxis.set_major_formatter(FuncFormatter(_yfmt))
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(abc["category"], abc["cum_pct"] * 100, color=PALETTE[3], marker="o", lw=2)
    ax2.axhline(80, ls="--", color="grey"); ax2.set_ylabel("累计占比(%)")
    ax2.set_ylim(0, 105)
    ax1.set_title("图9 销售额帕累托分析（ABC分类，组合图）", fontsize=13, weight="bold")
    save(fig, "fig09_帕累托_组合图.png")

    # ---------- 业务图 10：波士顿矩阵气泡图 ----------
    bcg = pd.read_csv(os.path.join(TABLE_DIR, "bcg_matrix.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"明星类(Star)": PALETTE[2], "现金牛(Cash Cow)": PALETTE[0],
              "问题类(Question)": PALETTE[1], "瘦狗类(Dog)": PALETTE[4]}
    for q, g in bcg.groupby("quadrant"):
        ax.scatter(g["rel_share"], g["growth"] * 100, s=g["revenue"] / bcg["revenue"].max() * 1500 + 80,
                   color=colors.get(q, "#999"), alpha=.6, label=q)
        for _, r in g.iterrows():
            ax.annotate(r["category"], (r["rel_share"], r["growth"] * 100), fontsize=7, ha="center")
    ax.axhline(bcg["growth"].median() * 100, ls="--", color="grey")
    ax.axvline(bcg["rel_share"].median(), ls="--", color="grey")
    ax.set_xlabel("相对市场份额"); ax.set_ylabel("增长率(%)")
    ax.set_title("图10 类目波士顿矩阵（BCG气泡图）", fontsize=13, weight="bold")
    ax.legend(fontsize=8)
    save(fig, "fig10_波士顿矩阵_气泡图.png")

    # ---------- 业务图 11：关联规则 Top 条形 ----------
    ar = safe_read("association_rules.csv")
    if ar is not None and len(ar):
        top = ar.sort_values("lift", ascending=True).tail(10).copy()
        top["rule"] = top["antecedents"] + " → " + top["consequents"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top["rule"], top["lift"], color=PALETTE[5])
        ax.set_title("图11 关联规则 Top10（按提升度，条形图）", fontsize=13, weight="bold")
        ax.set_xlabel("提升度(lift)")
        save(fig, "fig11_关联规则_条形图.png")

    # ---------- 业务图 12：Holt-Winters 预测带置信区间 ----------
    fc = safe_read("holt_winters_forecast.csv")
    if fc is not None and len(fc):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        actual = fc[fc["type"] == "actual"]; fore = fc[fc["type"] == "forecast"]
        ax.plot(actual["month"], actual["value"], color=PALETTE[0], marker="o", label="实际")
        ax.plot(fore["month"], fore["value"], color=PALETTE[3], marker="o", label="预测")
        ax.fill_between(fore["month"], fore["lower"], fore["upper"], color=PALETTE[3], alpha=.2, label="95%置信区间")
        ax.set_title("图12 周度销售额 Holt-Winters 预测（4周季节项，含95%置信区间）", fontsize=13, weight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(_yfmt)); ax.legend(); ax.grid(alpha=.3)
        ax.tick_params(axis="x", rotation=45)
        save(fig, "fig12_预测_置信区间图.png")

    # ---------- 业务图 13：RFM 八分类条形 ----------
    seg = pd.read_csv(os.path.join(TABLE_DIR, "rfm_segments.csv")).sort_values("customers", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(seg["segment"], seg["customers"], color=PALETTE[2])
    ax.set_title("图13 RFM 客户八分类（条形图）", fontsize=13, weight="bold")
    ax.set_xlabel("客户数")
    save(fig, "fig13_RFM八分类_条形图.png")

    # ---------- 业务图 14：同期群留存热力图 ----------
    coh = pd.read_csv(os.path.join(TABLE_DIR, "cohort_retention.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(coh.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(coh.columns))); ax.set_xticklabels(coh.columns, fontsize=7)
    ax.set_yticks(range(len(coh.index))); ax.set_yticklabels(coh.index, fontsize=7)
    ax.set_title("图14 同期群留存率热力图", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, fraction=.02, pad=.02)
    save(fig, "fig14_同期群_热力图.png")

    # ---------- 业务图 15：K-Means 分群散点 ----------
    prof = safe_read("kmeans_profile.csv")
    if prof is not None and "cluster" in rfm.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in sorted(rfm["cluster"].unique()):
            g = rfm[rfm["cluster"] == c]
            ax.scatter(g["F_score"], g["M_score"], s=8, alpha=.4, label=f"簇{c}")
        ax.set_xlabel("消费频次得分"); ax.set_ylabel("消费金额得分")
        ax.set_title("图15 K-Means 客户分群（散点图）", fontsize=13, weight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        save(fig, "fig15_KMeans分群_散点图.png")

    # ---------- 业务图 16：情感分析柱状 ----------
    if "sentiment" in reviews.columns:
        sl = reviews["label"].value_counts()
        order = [x for x in ["正面", "中性", "负面"] if x in sl.index]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(order, [sl[o] for o in order], color=[PALETTE[2], PALETTE[5], PALETTE[3]])
        ax.set_title("图16 评论情感分布（柱状图）", fontsize=13, weight="bold")
        ax.set_ylabel("评论数")
        save(fig, "fig16_情感分析_柱状图.png")

    # ---------- 业务图 17：转化漏斗（全部由真实交易明细派生） ----------
    # 口径：交易明细行 → 唯一发票 → 成交客户 → 复购客户(≥2 张发票) → 高价值客户(RFM 中 M 高于中位数)
    real = fact[fact["is_return"] == 0]
    n_lines = len(real)
    n_invoice = real["order_id"].nunique()
    n_buyer = real["customer_id"].nunique()
    inv_per_cust = real.groupby("customer_id")["order_id"].nunique()
    n_repeat = int((inv_per_cust >= 2).sum())
    high_m = rfm[rfm["Monetary"] >= rfm["Monetary"].median()] if "Monetary" in rfm.columns else rfm.iloc[0:0]
    n_high = len(high_m)
    funnel = [("交易明细行", n_lines), ("有效发票", n_invoice), ("成交客户", n_buyer),
              ("复购客户", n_repeat), ("高价值客户", n_high)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ys = range(len(funnel))
    maxv = funnel[0][1] or 1
    for i, (name, v) in zip(ys, funnel):
        ax.barh(i, v / maxv * 100, color=PALETTE[i % len(PALETTE)])
        ax.text(v / maxv * 100 / 2, i, f"{name}: {v:,}", va="center", ha="center", color="white", fontsize=9)
    ax.set_yticks(list(ys)); ax.set_yticklabels([f[0] for f in funnel])
    ax.set_xlabel("相对首层比例(%)"); ax.invert_yaxis()
    ax.set_title("图17 交易转化漏斗（基于真实交易明细，漏斗图）", fontsize=13, weight="bold")
    save(fig, "fig17_转化漏斗_漏斗图.png")

    print("[04_visuals] 已生成图表：")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith(".png"):
            print("  -", f)
    print("[04_visuals] 输出目录:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
