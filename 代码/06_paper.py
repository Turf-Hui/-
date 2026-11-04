# -*- coding: utf-8 -*-
"""
06_paper.py —— 由「论文正文.md」生成期末大论文 Word（含图表嵌入）
==============================================================
读取 docs/论文正文.md（已撰写、结构完整的 Markdown），转换为规范 Word 文档，
并自动嵌入 outputs/figures 下由 04_visuals.py 生成的图表。
依赖：python-docx（可选；若未安装则跳过以纯净文本生成，不影响论文主体）。
用法：pip install python-docx && python scripts/06_paper.py
"""
from __future__ import annotations
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..")
DOC_OUT = os.path.join(BASE, "docs", "期末大论文.docx")
MD_SRC = os.path.join(BASE, "docs", "论文正文.md")
FIG = os.path.join(BASE, "outputs", "figures")

CJK = "宋体"
CJK_HEI = "黑体"


def build():
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
    except Exception as e:
        raise SystemExit(f"需要 python-docx：pip install python-docx（{e}）")

    def set_cjk(run, font=CJK):
        run.font.name = font
        run._element.rPr.rFonts.set(qn("w:eastAsia"), font)

    md = open(MD_SRC, encoding="utf-8").read()
    lines = md.splitlines()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = CJK; style.font.size = Pt(11)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)

    # 图注行：提取 outputs/figures/xxx.png（可能被反引号包裹）与 **图X 描述**
    path_re = re.compile(r"`?(outputs/figures/[^`\s]+\.png)`?")
    cap_re = re.compile(r"\*\*(图[^*]+?)\*\*")
    img_count = 0
    for ln in lines:
        s = ln.strip()
        if not s or s == "---":
            continue
        m_path = path_re.search(s)
        fig_path = m_path.group(1).strip() if m_path else None
        m_cap = cap_re.search(s)
        fig_caption = m_cap.group(1).strip() if m_cap else None

        # 图注行：渲染图题（居中）+ 嵌入图片
        if fig_path:
            if fig_caption:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(fig_caption); r.bold = True; r.font.size = Pt(10); set_cjk(r, CJK_HEI)
            abspath = os.path.join(BASE, fig_path)
            if os.path.exists(abspath):
                pic_p = doc.add_paragraph(); pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pic_p.add_run().add_picture(abspath, width=Inches(6.0))
                img_count += 1
            else:
                print(f"[06_paper] 图片缺失（跳过）：{abspath}")
            continue

        if s.startswith("# "):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(s[2:]); r.bold = True; r.font.size = Pt(18); set_cjk(r, CJK_HEI)
        elif s.startswith("## "):
            p = doc.add_paragraph(); r = p.add_run(s[3:]); r.bold = True; r.font.size = Pt(14); set_cjk(r, CJK_HEI)
            p.paragraph_format.space_before = Pt(10)
        elif s.startswith("### "):
            p = doc.add_paragraph(); r = p.add_run(s[4:]); r.bold = True; r.font.size = Pt(12); set_cjk(r, CJK_HEI)
        elif s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet"); r = p.add_run(s[2:]); r.font.size = Pt(11); set_cjk(r)
        else:
            p = doc.add_paragraph()
            # 行内加粗 **x**
            parts = re.split(r"\*\*(.+?)\*\*", s)
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    r = p.add_run(part); r.bold = True; r.font.size = Pt(11); set_cjk(r)
                else:
                    r = p.add_run(part); r.font.size = Pt(11); set_cjk(r)
            p.paragraph_format.line_spacing = 1.5

    doc.save(DOC_OUT)
    print(f"[06_paper] 论文已生成（含图表嵌入 {img_count} 张）: {os.path.abspath(DOC_OUT)}")


if __name__ == "__main__":
    build()
