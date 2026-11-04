# 商业数据分析期末大作业

> **主数据源为课程附件 A 白名单内的 UCI Online Retail 真实公开数据集**


## 一、项目简介

本项目以 UCI Machine Learning Repository 的 **Online Retail** 在线零售交易数据集（英国非实体店礼品电商，2010-12-01 ~ 2011-12-09）为分析对象，综合运用 Python 数据合规获取、清洗、商业分析模型、数据可视化与交互式大屏，完成端到端商业数据分析。

- **分析对象**：真实交易明细（发票 / 商品 / 客户 / 国家 / 数量 / 单价 / 日期）
- **覆盖模型**：RFM 八分类、同期群留存、帕累托/ABC、波士顿矩阵、关联规则 Apriori、Holt-Winters 时间序列预测、K-Means 客户分群、中文评论情感分析（共 8 种，≥3 达标）
- **可视化**：17 张图表（8 种模板复现 + 8 张业务模型图 + 1 张转化漏斗）+ 1 个 Pyecharts 交互大屏（≥6 组件）
- **技术栈**：Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Statsmodels · MLxtend · Faker（仅补充）· Pyecharts · openpyxl
- **计量单位**：英镑（sterling，UCI 官方字段 `UnitPrice` 单位）

## 二、数据来源与合规声明

> 完整声明见 [`docs/合规声明.md`](docs/合规声明.md)，论文第 2 章亦作说明。

| 项目 | 内容 |
|---|---|
| **数据源名称** | **Online Retail**（UCI Machine Learning Repository, Dataset 352）——课程**附件 A** 白名单「UCI」条目 |
| **官方 URL** | https://archive.ics.uci.edu/dataset/352/online+retail |
| **官方文件下载 URL** | https://archive.ics.uci.edu/static/public/352/online+retail.zip |
| **获取方式** | 数据源官方提供的一次性下载文件（ZIP / XLSX）——白名单方式，**非爬虫** |
| **获取日期** | 2026-11-04 |
| **授权 / License** | UCI ML Repository 公开学术数据集，供教学/研究使用，须按官方 Citation Policy 标注引用 |
| **官方引用** | Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. |
| **原始规模** | 541,909 行 × 8 字段，2010-12-01 ~ 2011-12-09 |

**非爬虫声明**：仅对上述一个官方文件地址执行一次 `curl` 下载；未使用 Scrapy / Selenium / `requests+网页解析` / 模拟登录，未解析 HTML、未遍历页面。原始文件留存 `data/raw/` 供复核。

**抽样方法与随机种子**：以「发票 `InvoiceNo`」为抽样单元做等概率无放回随机抽样（保留被抽中发票的全部明细行，避免破坏购物篮共现结构），`RANDOM_SEED = 42`（`numpy.random.RandomState(42)`），从 18,402 张有效发票中抽取 `SAMPLE_INVOICES = 1500` 张，得到 **33,496 行明细**，清洗去重后主表 **33,123 行**（客户 1,049 · 商品 2,919 · 补充评论 18,357），净销售额 702,387.29 英镑，远超"主表不少于 500 条"要求。实际规模见 `data/raw/data_overview.json` 与 `data/clean/clean_report.json`。

**补充生成数据（按课程 2.1(3) 单独标注）**：UCI 原始数据集不含客户画像、商品类目、成本、渠道、评论文本，故用 Faker / 规则派生补充以下字段，且**核心交易度量（发票号、商品码、客户号、日期、数量、单价、金额、国家）全部为真实数据**：

| 补充字段 | 方式 |
|---|---|
| `gender` / `age` / `member_level` / `name` | Faker 按 Schema 生成【补充】 |
| `category` / `sub_category` | 由真实 `Description` 关键词规则派生【补充】 |
| `brand` / `channel` / `payment` | Faker 生成【补充】 |
| `cost` | 按 40% 毛利率估算（0.6 × 真实单价中位数）【补充估算】 |
| **整张 `fact_reviews`** | Faker 生成中文评论，仅用于情感分析模型【补充】 |

标注位置：`01_load.py` 行内注释、`data_overview.json` 的 `supplement_fields`、本表、`docs/合规声明.md` 第 4 节、论文 2.1/2.4 节。

## 三、目录结构

```
business-data-analysis-UCI/
├── data/
│   ├── raw/            # UCI 官方 XLSX 原文件 OnlineRetail.xlsx（23.7 MB）
│   │                  # + 四张中间 CSV + data_overview.json（含真实/补充字段清单）
│   └── clean/          # 星型模型 CSV / JSON 样例 / SQLite shop.db / 清洗报告
├── scripts/
│   ├── 01_load.py      # 数据获取：UCI 官方文件 + 抽样 + 补充标注
│   ├── 02_clean.py     # 清洗治理（缺失/异常/重复/类型日期/主外键对齐）
│   ├── 03_analysis.py  # 8 种商业分析模型
│   ├── 04_visuals.py   # 17 张图（150dpi，中文字体，统一命名）
│   ├── 05_dashboard.py # Pyecharts 交互大屏
│   ├── 06_paper.py     # 由论文正文.md 生成 Word（自动嵌图）
│   └── run_all.py      # 一键复现入口
├── 代码/               # 全套源码交付副本（与 scripts/ 逐字节一致，同样可直接运行）
│   ├── 01_load.py ~ 06_paper.py
│   ├── run_all.py
│   └── requirements.txt
├── outputs/
│   ├── figures/        # fig01–fig17 共 17 张 PNG
│   ├── tables/         # 模型结果表 CSV
│   ├── dashboard.html  # 交互式大屏（CDN 版，需联网）
│   ├── dashboard_离线版.html  # 单文件离线版，双击即开、断网可用（推荐提交）
│   └── assets/         # 离线版静态资源缓存，随项目交付，复现时无需联网
├── docs/
│   ├── 期末大论文_课程模板版.docx  # **最终提交版**（课程模板格式，含封面/评分表，18 图）
│   ├── 期末大论文.docx  # 06_paper.py 由 Markdown 自动生成的可复现版（17 图）
│   ├── 论文正文.md     # 论文 Markdown 源（论文改动以课程模板版为最终准绳）
│   └── 合规声明.md     # 数据来源合规声明
├── requirements.txt
└── README.md
```

## 四、复现步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键复现（获取→清洗→8模型→17图→大屏→论文 Word）
python scripts/run_all.py
#    若只需单独重生成论文 Word：python scripts/06_paper.py

# 3. 查看大屏：**双击 outputs/dashboard_离线版.html**（推荐，单文件、断网可用）
#    outputs/dashboard.html 为 CDN 版，需联网加载 echarts；两者组件与数据完全相同

# 说明：代码/ 为源码交付副本，内容与 scripts/ 完全一致，同样可直接运行
```

数据获取环节：`01_load.py` 若发现 `data/raw/` 缺少完整官方文件，会自动从上述 UCI 官方 URL 下载（非爬虫）。由于该服务器响应较慢且忽略 `Range` 请求，脚本内置了**断点续传式流式下载**：利用 UCI ZIP 为 store（未压缩）模式的特性，边下载边把 ZIP 内 `Online Retail.xlsx` 的原始字节直接写出，连接中断后跳过已写入字节继续追加，进度不会倒退，因此慢速网络下也能可靠完成。离线场景可预先把 `OnlineRetail.xlsx` 或 `online_retail.zip` 放入 `data/raw/`。

## 五、图表索引（fig01–fig17 + 大屏）

| 编号 | 文件名 | 类型 | 归属 |
|---|---|---|---|
| fig01 | 销售趋势_折线图 | 折线 | 模板复现 |
| fig02 | 类目占比_环形图 | 环形 | 模板复现 |
| fig03 | 地区分布_柱状图（国家/地区 Top15） | 柱状 | 模板复现 |
| fig04 | 渠道占比_堆叠柱 | 堆叠柱 | 模板复现 |
| fig05 | 会员等级_条形图 | 条形 | 模板复现 |
| fig06 | 价格分布_直方图 | 直方图 | 模板复现 |
| fig07 | 客单价_箱线图 | 箱线 | 模板复现 |
| fig08 | 类目月份_热力图 | 矩阵热力 | 模板复现 |
| fig09 | 帕累托_组合图 | 组合 | 业务(帕累托) |
| fig10 | 波士顿矩阵_气泡图 | 气泡 | 业务(BCG) |
| fig11 | 关联规则_条形图 | 条形 | 业务(Apriori) |
| fig12 | 预测_置信区间图（周度 Holt-Winters） | 折线+CI | 业务(预测) |
| fig13 | RFM八分类_条形图 | 条形 | 业务(RFM) |
| fig14 | 同期群_热力图 | 热力 | 业务(同期群) |
| fig15 | KMeans分群_散点图 | 散点 | 业务(K-Means) |
| fig16 | 情感分析_柱状图 | 柱状 | 业务(情感·补充数据) |
| fig17 | 转化漏斗_漏斗图 | 漏斗 | 业务(漏斗) |

> 注：fig16 情感分析基于【补充生成】的中文评论（UCI 原数据无评论字段），已在论文与合规声明中单独标注。
---

© 2026 商业数据分析课程期末大作业 · UCI Online Retail 
