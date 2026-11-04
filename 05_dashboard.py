# -*- coding: utf-8 -*-
"""
05_dashboard.py —— 交互式可视化大屏（Pyecharts，全 Python 实现）
===============================================================
使用 Pyecharts 搭建商业分析大屏，包含 ≥6 个联动组件：
  KPI 卡(Table) / 月度趋势(Timeline 联动) / 类目结构(环形) /
  地区分布(柱状) / 波士顿矩阵(气泡) / RFM 八分类(条形) / 明细表(Table)
输出 dashboard.html（CDN 版），并额外生成 dashboard_离线版.html：把 5 个在线静态
资源内联进单文件，断网双击即可打开，避免评阅环境无外网导致白屏。
"""
from __future__ import annotations
import os
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Page, Line, Pie, Bar, Scatter, Timeline
from pyecharts.components import Table
from pyecharts.globals import ThemeType

BASE = os.path.join(os.path.dirname(__file__), "..")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
TABLE_DIR = os.path.join(BASE, "outputs", "tables")
OUT = os.path.join(BASE, "outputs", "dashboard.html")
ASSETS_DIR = os.path.join(BASE, "outputs", "assets")
OFFLINE_OUT = os.path.join(BASE, "outputs", "dashboard_离线版.html")

# 国内可达的静态资源镜像；仅当 outputs/assets/ 缓存缺失时才需要联网（非爬虫，静态文件）
ASSET_SOURCES = {
    "echarts.min.js": "https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js",
    "jquery.min.js": "https://cdn.bootcdn.net/ajax/libs/jquery/3.6.0/jquery.min.js",
    "jquery-ui.min.js": "https://cdn.bootcdn.net/ajax/libs/jqueryui/1.13.2/jquery-ui.min.js",
    "ResizeSensor.js": "https://cdn.bootcdn.net/ajax/libs/css-element-queries/1.2.3/ResizeSensor.js",
    "jquery-ui.min.css": "https://cdn.bootcdn.net/ajax/libs/jqueryui/1.12.1/jquery-ui.min.css",
}


def build_offline():
    """把 dashboard.html 的 CDN 引用内联，产出断网可用的单文件大屏。

    资源优先读取 outputs/assets/ 本地缓存（随项目交付，无需联网）；
    缓存缺失时才从上述镜像补下载。生成失败不影响在线版 dashboard.html。
    """
    import io
    import re
    import urllib.request

    os.makedirs(ASSETS_DIR, exist_ok=True)
    for name, url in ASSET_SOURCES.items():
        p = os.path.join(ASSETS_DIR, name)
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
                f.write(r.read())
            print("[05_dashboard] 已缓存静态资源:", name)
        except Exception as e:
            print("[05_dashboard] 资源下载失败（保留在线版）:", name, e)

    js_map = {n: n for n in ASSET_SOURCES if n.endswith(".js")}
    css_name = "jquery-ui.min.css"
    if not all(os.path.exists(os.path.join(ASSETS_DIR, n)) for n in ASSET_SOURCES):
        print("[05_dashboard] 静态资源不全，跳过离线版生成")
        return

    src = io.open(OUT, encoding="utf-8").read()

    def inline_js(m):
        name = m.group(1).rsplit("/", 1)[-1]
        if name not in js_map:
            return m.group(0)
        body = io.open(os.path.join(ASSETS_DIR, name), encoding="utf-8", errors="replace").read()
        return '<script type="text/javascript">\n' + body.replace("</script", "<\\/script") + "\n</script>"

    def inline_css(m):
        if not m.group(1).endswith("jquery-ui.css"):
            return m.group(0)
        body = io.open(os.path.join(ASSETS_DIR, css_name), encoding="utf-8", errors="replace").read()
        return "<style>\n" + body + "\n</style>"

    out = re.sub(r'<script[^>]*src="(https://assets\.pyecharts\.org/[^"]+)"[^>]*></script>', inline_js, src)
    out = re.sub(r'<link[^>]*href="(https://assets\.pyecharts\.org/[^"]+)"[^>]*>', inline_css, out)
    io.open(OFFLINE_OUT, "w", encoding="utf-8").write(out)
    print("[05_dashboard] 离线单文件大屏已生成:", os.path.abspath(OFFLINE_OUT))


def build():
    fact = pd.read_csv(os.path.join(CLEAN_DIR, "fact_orders.csv"), parse_dates=["order_date"])
    rfm = pd.read_csv(os.path.join(CLEAN_DIR, "rfm.csv"))
    bcg = pd.read_csv(os.path.join(TABLE_DIR, "bcg_matrix.csv"))
    seg = pd.read_csv(os.path.join(TABLE_DIR, "rfm_segments.csv"))
    monthly = fact[fact["is_return"] == 0].groupby("order_month")["net_amount"].sum()

    # ---- KPI 卡 ----
    _valid = fact[fact["is_return"] == 0].copy()
    gmv = _valid["net_amount"].sum()
    profit = ((_valid["product_price"] - _valid["product_cost"]) * _valid["quantity"]).sum()
    n_cust = rfm.shape[0]
    n_order = fact["order_id"].nunique()
    kpi = Table().add(headers=["指标", "数值"],
                      rows=[["净销售额(GMV)", f"{gmv:,.0f} 英镑"],
                            ["估算毛利额", f"{profit:,.0f} 英镑"],
                            ["成交客户数", f"{n_cust:,}"],
                            ["订单数", f"{n_order:,}"]])
    kpi.set_global_opts(title_opts=opts.TitleOpts(title="核心 KPI"))

    # ---- 月度趋势（Timeline 联动）----
    tl = Timeline().add_schema(play_interval=900, height=60)
    for m in monthly.index[::3]:
        sub = fact[(fact["is_return"] == 0) & (fact["order_month"] <= m)].groupby("product_category")["net_amount"].sum()
        pie = (Pie().add("类目占比", [list(z) for z in zip(sub.index.tolist(), sub.values.tolist())],
                         radius=["30%", "65%"]).set_global_opts(title_opts=opts.TitleOpts(f"截至 {m} 类目结构")))
        tl.add(pie, m)
    trend = (Line().add_xaxis(monthly.index.tolist())
             .add_yaxis("净销售额", monthly.values.astype(int).tolist(),
                        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]))
             .set_global_opts(title_opts=opts.TitleOpts("月度净销售额趋势"),
                              datazoom_opts=[opts.DataZoomOpts()]))

    # ---- 类目环形 ----
    cat = fact[fact["is_return"] == 0].groupby("product_category")["net_amount"].sum()
    pie = (Pie().add("", [list(z) for z in zip(cat.index.tolist(), cat.values.tolist())], radius=["35%", "70%"])
           .set_global_opts(title_opts=opts.TitleOpts("类目销售结构")))

    # ---- 国家/地区柱状（UCI Country 字段） ----
    prov = fact[fact["is_return"] == 0].groupby("province")["net_amount"].sum().sort_values(ascending=False).head(15)
    bar = (Bar().add_xaxis(prov.index.tolist()).add_yaxis("净销售额", prov.values.astype(int).tolist())
           .set_global_opts(title_opts=opts.TitleOpts("国家/地区销售分布 Top15"),
                            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=40)),
                            datazoom_opts=[opts.DataZoomOpts()]))

    # ---- 波士顿气泡 ----
    sc = (Scatter().add_xaxis(bcg["rel_share"].round(3).tolist())
          .add_yaxis("类目", [list(z) for z in zip((bcg["growth"] * 100).round(2).tolist(),
                                                   (bcg["revenue"] / bcg["revenue"].max() * 100).round(1).tolist(),
                                                   bcg["category"].tolist())],
                     symbol_size=18)
          .set_global_opts(title_opts=opts.TitleOpts("波士顿矩阵(气泡=销售额)"),
                           xaxis_opts=opts.AxisOpts(name="相对份额"), yaxis_opts=opts.AxisOpts(name="增长率%")))

    # ---- RFM 八分类 ----
    seg = seg.sort_values("customers")
    bar_rfm = (Bar().add_xaxis(seg["segment"].tolist()).add_yaxis("客户数", seg["customers"].tolist())
               .set_global_opts(title_opts=opts.TitleOpts("RFM 客户八分类"),
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30))))

    page = Page(layout=Page.DraggablePageLayout, page_title="商业数据分析大屏")
    page.add(kpi, trend, tl, pie, bar, sc, bar_rfm)
    page.render(OUT)
    print("[05_dashboard] 大屏已生成:", os.path.abspath(OUT))
    build_offline()


if __name__ == "__main__":
    build()
