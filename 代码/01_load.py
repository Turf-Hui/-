# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import random
import re
import zipfile
import numpy as np
import pandas as pd
from faker import Faker

# ============================ 全局配置 ============================
RANDOM_SEED = 42
SAMPLE_INVOICES = 1500        # 抽样：保留 1500 张发票的完整明细（约 2–3 万行）
USE_REAL_SOURCE = True        # 主数据源 = 附件 A 的 UCI Online Retail

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_XLSX_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"
FETCH_DATE = "2026-11-04"

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
RAW_ZIP = os.path.join(RAW_DIR, "online_retail.zip")

fake = Faker("zh_CN")
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 品类关键词规则（基于真实 Description 文本派生，属"补充维度"）
CATEGORY_RULES = [
    (r"christmas|easter|party|birthday|advent|tree|bauble|ornament", "节庆布置"),
    (r"bag|pouch|rucksack|backpack|tote|suitcase|purse", "箱包收纳"),
    (r"mug|cup|bowl|plate|jug|glass|bottle|teapot|tray|dish|cutlery|spoon|fork", "餐厨器皿"),
    (r"light|lamp|bulb|lantern|candle|holder|chandelier", "灯饰照明"),
    (r"card|wrap|gift|tag|label|sticker|sign|stationery|notebook|pen|pencil", "文具礼品"),
    (r"t-light|heart|love|vintage|retro|decor|frame|mirror|hook|hanging|clock|vase|photo", "家居装饰"),
    (r"t-shirt|shirt|dress|jacket|trouser|jumper|cardigan|knit|sock|hat|scarf|apron|glove", "服饰针织"),
    (r"baby|child|kid|boy|girl|toy|game|play|teddy|doll|puzzle|building block", "母婴玩具"),
    (r"garden|flower|plant|bird|dog|cat|pet|watering|seed|pot", "园艺宠物"),
    (r"chocolate|cake|biscuit|cookie|tea|coffee|jam|sweet|spice|sugar|food", "食品饮料"),
    (r"soap|bath|cream|towel|perfume|cosmetic|nail|lip|beauty", "个护美妆"),
    (r"bike|cycle|sport|gym|fitness|yoga|outdoor|camping", "运动户外"),
    (r"book|kit|set|tool|paint|draw|craft|sewing|ribbon|bead|felt", "手工文创"),
]
DEFAULT_CATEGORY = "其他日用"
BRANDS = ["UCI", "Britannia", "Manor", "Cottage", "Harbour", "Vintage", "Rosewood", "Kingdom", "Willow", "Thames"]
CHANNELS = ["官网", "APP", "第三方平台", "电话/批发订购"]
PAYMENTS = ["信用卡", "PayPal", "银行转账", "礼品卡"]


def _map_category(desc: str) -> str:
    if not isinstance(desc, str):
        return DEFAULT_CATEGORY
    d = desc.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, d):
            return cat
    return DEFAULT_CATEGORY


# ============================ 官方文件读取（含缺文件时的稳健续传下载） ============================
def _valid_xlsx(path: str) -> bool:
    """校验 xlsx 是否完整可读（xlsx 本质是 ZIP，需含完整结尾目录）。"""
    return os.path.exists(path) and zipfile.is_zipfile(path)


def _read_exact(r, n):
    buf = b""
    while len(buf) < n:
        b = r.read(n - len(buf))
        if not b:
            raise IOError("stream ended while reading ZIP header")
        buf += b
    return buf


def _skip(r, n):
    while n > 0:
        b = r.read(min(1 << 20, n))
        if not b:
            raise IOError("stream ended while skipping")
        n -= len(b)


def _download_uci_stream(xlsx_path: str, max_retries: int = 30):
    """稳健下载（非爬虫，仅对一个官方文件地址发起一次性文件下载）。

    UCI 的 online+retail.zip 采用 store（未压缩）模式，ZIP 内首个文件
    `Online Retail.xlsx` 的原始字节自 data_offset 起连续存放，因而可以边下载
    边把这段字节直接写成 xlsx，**无需等待 ZIP 尾部目录**。

    该服务器忽略 Range 请求（返回 200 而非 206），且慢速网络下连接易中断。
    本函数每次重连后跳过 (data_offset + 已写入字节数) 再追加写入，
    保证断点续传、进度绝不倒退——这是普通 urlretrieve / curl --retry 做不到的。
    """
    import urllib.request
    import struct
    import time

    target = None
    for attempt in range(1, max_retries + 1):
        have = os.path.getsize(xlsx_path) if os.path.exists(xlsx_path) else 0
        if target is not None and have >= target:
            return
        try:
            req = urllib.request.Request(
                UCI_ZIP_URL,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
            r = urllib.request.urlopen(req, timeout=120)
            head = _read_exact(r, 30)
            if head[:4] != b"PK\x03\x04":
                raise IOError("响应不是 ZIP 本地文件头")
            ver, flag, method, tm, dt, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", head[4:30])
            _read_exact(r, nlen)
            if elen:
                _skip(r, elen)
            data_offset = 30 + nlen + elen
            if method != 0:
                # 非 store 模式：退回"下载完整 ZIP 再解压"的普通路径
                r.close()
                raise RuntimeError("ZIP 非 store 模式，改用整包下载")
            target = csize
            if have >= target:
                r.close()
                return
            # 上面已用 _read_exact 消耗掉 (30 + nlen + elen) 字节的外层 ZIP 头，
            # 因此这里只需再跳过本轮续传已写入的 have 字节，切勿重复跳 data_offset。
            _skip(r, have)
            with open(xlsx_path, "ab") as f:
                remaining = target - have
                while remaining > 0:
                    b = r.read(min(1 << 20, remaining))
                    if not b:
                        raise IOError("stream ended during body")
                    f.write(b); f.flush()
                    remaining -= len(b)
            r.close()
            print(f"[01_load] 官方文件下载完成：{os.path.getsize(xlsx_path):,} 字节")
            return
        except RuntimeError:
            # 整包兜底
            print("[01_load] 改用整包下载 ...")
            urllib.request.urlretrieve(UCI_ZIP_URL, RAW_ZIP)
            if zipfile.is_zipfile(RAW_ZIP):
                with zipfile.ZipFile(RAW_ZIP) as z:
                    names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))]
                    t = sorted(names, key=lambda n: (not n.lower().endswith(".xlsx"), n))[0]
                    z.extract(t, RAW_DIR)
                    if os.path.abspath(os.path.join(RAW_DIR, t)) != os.path.abspath(xlsx_path):
                        os.replace(os.path.join(RAW_DIR, t), xlsx_path)
            return
        except Exception as e:
            have = os.path.getsize(xlsx_path) if os.path.exists(xlsx_path) else 0
            print(f"[01_load] 连接中断({e})，已保留 {have/1048576:.2f} MB，{attempt}/{max_retries} 续传重试 ...")
            time.sleep(2)
    raise RuntimeError("官方文件下载失败，请手动下载后放入 data/raw/OnlineRetail.xlsx，详见 README")


def _ensure_source_file() -> str:
    """返回可用的 UCI 数据文件路径；本地缺失/不完整时自动从官方地址续传下载（非爬虫）。"""
    xlsx = os.path.join(RAW_DIR, "OnlineRetail.xlsx")
    # 优先复用任何已完整落地的官方文件（兼容手动放置的其他命名）
    for f in os.listdir(RAW_DIR):
        p = os.path.join(RAW_DIR, f)
        if f.lower().endswith((".xlsx", ".xls", ".csv")) and "retail" in f.lower():
            if f.lower().endswith(".csv"):
                return p
            if _valid_xlsx(p):
                return p
    if not _valid_xlsx(xlsx):
        print("[01_load] 本地无完整官方文件，从 UCI 官方一次性下载（非爬虫，支持断点续传）...")
        _download_uci_stream(xlsx)
    return xlsx


def _read_uci(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        for enc in ("utf-8", "iso-8859-1", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(path)


# ============================ 主数据源加载 ============================
def load_uci() -> dict:
    src = _ensure_source_file()
    print("[01_load] 读取 UCI Online Retail 官方文件:", src)
    df = _read_uci(src)
    n_raw = len(df)
    print(f"[01_load] 原始行数: {n_raw:,} | 字段: {list(df.columns)}")

    # ---- 基础治理（获取阶段先剔除明显无效记录）----
    df.columns = [str(c).strip() for c in df.columns]
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()
    for c in ["Quantity", "UnitPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]      # 剔除取消单
    df = df[df["Quantity"] > 0]                                     # 剔除退货/负数
    df = df[df["UnitPrice"] > 0]                                    # 剔除免费/异常单价
    df = df[df["StockCode"].str.match(r"^\d{5}")]                   # 剔除 POST/DOT/M 等非商品费用码
    df = df.drop_duplicates()                                       # 重复值去除
    df["Amount"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    n_valid = len(df)

    # ---- 抽样：以发票为单元、无放回等概率、种子 42，保持订单明细完整 ----
    rnd = np.random.RandomState(RANDOM_SEED)
    inv_ids = df["InvoiceNo"].unique()
    n_invoice_total = len(inv_ids)
    keep_n = min(SAMPLE_INVOICES, n_invoice_total)
    keep = rnd.choice(inv_ids, keep_n, replace=False)
    df = df[df["InvoiceNo"].isin(keep)].copy()
    n_sampled = len(df)

    # ---- 字段标准化 ----
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["order_date"] = df["InvoiceDate"].dt.date.astype(str)
    df["quantity"] = df["Quantity"].astype(int)
    df["unit_price"] = df["UnitPrice"].round(2)
    df["amount"] = df["Amount"]
    df["is_return"] = 0

    # 客户维度：真实首单日期 + 真实主要国家
    cust_first = df.groupby("CustomerID")["InvoiceDate"].min().dt.date.astype(str)
    cust_country = df.groupby("CustomerID")["Country"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    # 商品维度：真实 Description / 真实单价中位数
    prod_desc = df.groupby("StockCode")["Description"].agg(
        lambda s: " ".join(dict.fromkeys([x for x in s if isinstance(x, str) and x]))[:80])
    prod_price = df.groupby("StockCode")["UnitPrice"].median().round(2)

    # ================= 客户维度表（真实 ID/国家/首单日 + Faker 补充画像）=================
    cust_rows = []
    for cid in cust_first.index:
        country = cust_country.get(cid, "United Kingdom")
        cust_rows.append({
            "customer_id": cid,
            "name": fake.name(),                                   # 【补充-Faker】
            "gender": random.choice(["男", "女"]),                   # 【补充-Faker】
            "age": int(np.clip(np.random.normal(40, 12), 18, 78)),  # 【补充-Faker】
            "province": country,                                   # 真实 Country（国家级地理）
            "city": country,                                       # 真实 Country
            "register_date": cust_first[cid],                     # 真实：窗口内首单日期
            "member_level": int(np.random.choice([1, 2, 3, 4, 5], p=[.35, .3, .2, .1, .05])),  # 【补充-Faker】
        })
    customers = pd.DataFrame(cust_rows)

    # ================= 商品维度表（真实 StockCode/Description/单价 + 补充派生）=================
    prod_rows = []
    for sc in prod_desc.index:
        desc = prod_desc[sc]
        price = float(prod_price.get(sc, 3.0))
        prod_rows.append({
            "product_id": sc,                                     # 真实 StockCode
            "category": _map_category(desc),                      # 【补充】由真实 Description 关键词派生
            "sub_category": (desc[:16] if desc else "未命名"),      # 【补充】真实描述截断
            "brand": random.choice(BRANDS),                        # 【补充-Faker】
            "cost": round(price * 0.6, 2),                          # 【补充估算】按 40% 毛利率反推成本
            "price": price,                                        # 真实：该商品单价中位数
        })
    products = pd.DataFrame(prod_rows)

    # ================= 交易明细（核心度量全部来自 UCI 真实数据）=================
    prod_lookup = products.set_index("product_id")[["category", "sub_category", "brand"]].to_dict("index")
    order_rows = []
    for r in df.itertuples(index=False):
        pl = prod_lookup.get(r.StockCode, {})
        order_rows.append({
            "order_id": r.InvoiceNo,                  # 真实 InvoiceNo
            "customer_id": r.CustomerID,              # 真实 CustomerID
            "product_id": r.StockCode,                # 真实 StockCode
            "category": pl.get("category", DEFAULT_CATEGORY),       # 【补充派生】
            "sub_category": pl.get("sub_category", "未命名"),         # 【补充派生】
            "brand": pl.get("brand", "UCI"),                         # 【补充派生】
            "province": r.Country,                    # 真实 Country
            "city": r.Country,                        # 真实 Country
            "channel": random.choice(CHANNELS),       # 【补充-Faker】
            "payment": random.choice(PAYMENTS),       # 【补充-Faker】
            "order_date": r.order_date,               # 真实 InvoiceDate
            "quantity": int(r.quantity),              # 真实 Quantity
            "unit_price": float(r.unit_price),        # 真实 UnitPrice
            "amount": float(r.amount),                # 真实 Quantity × UnitPrice
            "is_return": 0,
        })
    orders = pd.DataFrame(order_rows)

    # ================= 评论明细（Faker 补充，仅用于情感分析；单独标注）=================
    POS = ["物流很快，包装完好，非常满意！", "东西质量超出预期，性价比高，会回购。",
           "客服态度好，发货及时，体验很棒。", "用了一段时间，效果不错，推荐购买。",
           "外观精美，做工扎实，物超所值。", "比想象中好，家人都很喜欢。"]
    NEU = ["商品一般，没什么特别的感觉。", "发货速度正常，东西还行吧。",
           "价格适中，质量中规中矩。", "包装普通，整体还可以接受。"]
    NEG = ["物流太慢了，等了很久才到。", "收到货发现有破损，质量堪忧。",
           "和描述不符，有点失望。", "客服响应慢，售后体验差。",
           "用了没几天就坏了，不推荐。", "味道一般，性价比低。"]
    review_rows = []
    rnd_r = random.Random(RANDOM_SEED)
    for _, o in orders.iterrows():
        if rnd_r.random() > 0.55:      # 55% 的订单行产生评论
            continue
        rr = rnd_r.random()
        if rr < 0.7:
            rating, pool = 5, POS
        elif rr < 0.88:
            rating, pool = 3, NEU
        else:
            rating, pool = 1, NEG
        if rnd_r.random() < 0.06:
            rating = rnd_r.choice([1, 2, 4, 5])
        review_rows.append({
            "review_id": f"R{len(review_rows)+1:06d}",
            "order_id": o["order_id"],
            "customer_id": o["customer_id"],
            "product_id": o["product_id"],
            "rating": rating,
            "comment_text": rnd_r.choice(pool),
            "review_date": o["order_date"],
        })
    reviews = pd.DataFrame(review_rows)

    overview = {
        "data_source": "UCI Online Retail (UCI Machine Learning Repository, Dataset 352)",
        "source_url": UCI_PAGE_URL,
        "download_url": UCI_ZIP_URL,
        "fetch_method": "数据源官方一次性下载文件 ZIP/XLSX（白名单，非爬虫）",
        "fetch_date": FETCH_DATE,
        "license": "UCI ML Repository 公开数据集，教学/研究使用，须标注引用",
        "citation": "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.",
        "random_seed": RANDOM_SEED,
        "sample_method": f"以发票 InvoiceNo 为抽样单元等概率无放回抽样 {keep_n}/{n_invoice_total} 张发票，保留其全部明细行",
        "n_raw_rows": int(n_raw),
        "n_valid_rows": int(n_valid),
        "n_invoice_total": int(n_invoice_total),
        "n_invoice_sampled": int(keep_n),
        "n_sampled_rows": int(n_sampled),
        "n_customers": int(len(customers)),
        "n_products": int(len(products)),
        "n_order_lines": int(len(orders)),
        "n_orders": int(orders["order_id"].nunique()),
        "n_reviews_supplement": int(len(reviews)),
        "date_min": str(df["order_date"].min()),
        "date_max": str(df["order_date"].max()),
        "gmv": round(float(orders["amount"].sum()), 2),
        "real_fields": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
        "supplement_fields": ["gender", "age", "member_level", "category", "sub_category", "brand",
                              "cost", "channel", "payment", "fact_reviews(评论明细)"],
        "supplement_note": "以上字段 UCI 原始数据集不含，按课程 2.1(3) 用 Faker / 规则派生补充，已单独标注",
    }
    with open(os.path.join(RAW_DIR, "data_overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print("[01_load] 真实主源加载完成：")
    print(json.dumps({k: overview[k] for k in
                      ["n_raw_rows", "n_valid_rows", "n_invoice_sampled", "n_sampled_rows",
                       "n_customers", "n_products", "n_orders", "n_reviews_supplement", "date_min", "date_max", "gmv"]},
                     ensure_ascii=False, indent=2))
    return {"customers": customers, "products": products, "orders": orders, "reviews": reviews,
            "overview": overview}


def main():
    if not USE_REAL_SOURCE:
        raise SystemExit("本版本主数据源固定为附件 A 的 UCI Online Retail，USE_REAL_SOURCE 必须保持 True。")
    data = load_uci()
    customers, products, orders, reviews = (
        data["customers"], data["products"], data["orders"], data["reviews"])

    customers.to_csv(os.path.join(RAW_DIR, "customers.csv"), index=False, encoding="utf-8-sig")
    products.to_csv(os.path.join(RAW_DIR, "products.csv"), index=False, encoding="utf-8-sig")
    orders.to_csv(os.path.join(RAW_DIR, "orders.csv"), index=False, encoding="utf-8-sig")
    reviews.to_csv(os.path.join(RAW_DIR, "reviews.csv"), index=False, encoding="utf-8-sig")
    print("[01_load] 原始文件已写入:", os.path.abspath(RAW_DIR))


if __name__ == "__main__":
    main()
