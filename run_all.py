# -*- coding: utf-8 -*-
"""
run_all.py —— 一键复现入口
=========================
按顺序执行：数据获取 → 清洗 → 模型分析 → 可视化 → 大屏 → 论文 Word。
克隆仓库后只需：  pip install -r requirements.txt  &&  python scripts/run_all.py
数据获取环节会自动从 UCI 官方一次性下载地址补下载缺失文件（非爬虫）。
"""
from __future__ import annotations
import runpy
import os

SCRIPTS = ["01_load.py", "02_clean.py", "03_analysis.py",
           "04_visuals.py", "05_dashboard.py", "06_paper.py"]


def main():
    here = os.path.dirname(__file__)
    for s in SCRIPTS:
        print("\n" + "=" * 60)
        print(f">>> 运行 {s}")
        print("=" * 60)
        runpy.run_path(os.path.join(here, s), run_name="__main__")


if __name__ == "__main__":
    main()
