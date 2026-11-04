# -*- coding: utf-8 -*-
"""
02_clean.py —— 数据清洗与治理
"""
from __future__ import annotations
import os
import sqlite3
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE, "data", "raw")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)


def load_raw():
    customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"), dtype={"customer_id": str})
    products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"), dtype={"product_id": str})
    orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"), dtype={"order_id": str, "customer_id": str, "product_id": str})
    reviews = pd.read_csv(os.path.join(RAW_DIR, "reviews.csv"), dtype={"review_id": str, "order_id": str, "customer_id": str, "product_id": str})
    return customers, products, orders, reviews


def clean_orders(orders: pd.DataFrame):
    df = orders.copy()
    # 1) 类型与日期格式统一
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for c in ["unit_price", "amount", "quantity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_return"] = df["is_return"].fillna(0).astype(int)

    # 2) 缺失值处理：关键字段缺失直接剔除，并计数
    critical = ["order_id", "customer_id", "product_id", "order_date", "amount"]
    n_before = len(df)
    df = df.dropna(subset=critical)
    n_missing = n_before - len(df)

    # 3) 重复值去除：同一订单行完全重复
    n_dup = df.duplicated(subset=["order_id", "product_id", "customer_id", "order_date"]).sum()
    df = df.drop_duplicates(subset=["order_id", "product_id", "customer_id", "order_date"])

    # 4) 异常值识别（金额 IQR 法）：标记但不盲目删除，单独输出异常清单
    q1, q3 = df["amount"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_amount_outlier"] = ((df["amount"] < low) | (df["amount"] > high)).astype(int)
    outliers = df[df["is_amount_outlier"] == 1].copy()

    # 5) 派生指标
    df["net_amount"] = np.where(df["is_return"] == 1, 0.0, df["amount"])
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    return df, dict(n_missing=int(n_missing), n_dup=int(n_dup), n_outlier=int(len(outliers))), outliers


def align_foreign_keys(orders, customers, products):
    # 多表主外键对齐：剔除订单中不存在于维度表的客户/商品
    cids = set(customers["customer_id"])
    pids = set(products["product_id"])
    bad_cust = orders[~orders["customer_id"].isin(cids)]
    bad_prod = orders[~orders["product_id"].isin(pids)]
    orders = orders[orders["customer_id"].isin(cids) & orders["product_id"].isin(pids)]
    return orders, dict(bad_customer=int(len(bad_cust)), bad_product=int(len(bad_prod)))


def build_star_schema(orders, customers, products):
    # 事实表 + 维度表（星型模型），便于后续分析与复用
    fact = orders.merge(customers, on="customer_id", how="left", suffixes=("", "_cust")) \
                 .merge(products, on="product_id", how="left", suffixes=("", "_prod"))
    fact = fact[[
        "order_id", "customer_id", "product_id", "order_date", "order_month", "year", "month",
        "province", "city", "channel", "payment", "category", "sub_category", "brand",
        "quantity", "unit_price", "amount", "net_amount", "is_return", "is_amount_outlier",
        "gender", "age", "member_level", "cost", "price", "register_date",
    ]].rename(columns={"cost": "product_cost", "price": "product_price", "category": "product_category"})
    fact["profit"] = (fact["product_price"] - fact["product_cost"]) * fact["quantity"]
    fact["profit"] = fact["profit"].where(fact["is_return"] == 0, 0.0)
    return fact


def main():
    print("[02_clean] 读取原始数据 ...")
    customers, products, orders, reviews = load_raw()

    orders, clean_stat, outliers = clean_orders(orders)
    orders, fk_stat = align_foreign_keys(orders, customers, products)
    fact = build_star_schema(orders, customers, products)

    # 导出：CSV
    customers.to_csv(os.path.join(CLEAN_DIR, "dim_customers.csv"), index=False, encoding="utf-8-sig")
    products.to_csv(os.path.join(CLEAN_DIR, "dim_products.csv"), index=False, encoding="utf-8-sig")
    fact.to_csv(os.path.join(CLEAN_DIR, "fact_orders.csv"), index=False, encoding="utf-8-sig")
    reviews.to_csv(os.path.join(CLEAN_DIR, "fact_reviews.csv"), index=False, encoding="utf-8-sig")
    outliers.to_csv(os.path.join(CLEAN_DIR, "amount_outliers.csv"), index=False, encoding="utf-8-sig")

    # 导出：JSON（样例 + 全量可选）
    fact.head(200).to_json(os.path.join(CLEAN_DIR, "fact_orders_sample.json"), orient="records", force_ascii=False, indent=2)

    # 导出：SQLite
    db = os.path.join(CLEAN_DIR, "shop.db")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    customers.to_sql("dim_customers", con, if_exists="replace", index=False)
    products.to_sql("dim_products", con, if_exists="replace", index=False)
    fact.to_sql("fact_orders", con, if_exists="replace", index=False)
    reviews.to_sql("fact_reviews", con, if_exists="replace", index=False)
    con.close()

    stat = {
        "clean_stat": clean_stat,
        "fk_stat": fk_stat,
        "fact_rows": int(len(fact)),
        "gmv_net": round(float(fact["net_amount"].sum()), 2),
        "total_profit": round(float(fact["profit"].sum()), 2),
        "date_min": str(fact["order_date"].min().date()),
        "date_max": str(fact["order_date"].max().date()),
    }
    import json
    with open(os.path.join(CLEAN_DIR, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print("[02_clean] 清洗与导出完成：")
    print(json.dumps(stat, ensure_ascii=False, indent=2))
    print("[02_clean] 已导出 CSV / JSON / SQLite 至:", os.path.abspath(CLEAN_DIR))


if __name__ == "__main__":
    main()
