# -*- coding: utf-8 -*-
"""
step1_load.py  ——  第 1~2 天用：把资料读出来，切成小块，存成 chunks.json

做什么：
  1. 读 data/ 目录下的 PDF / Word / 网页另存的 txt
  2. 按 3 种策略切分成 chunk（小块文本）
  3. 存到 chunks.json，供下一步向量化

怎么用：
  python step1_load.py            # 默认用 fixed（固定长度）策略
  python step1_load.py section    # 按章节切
  python step1_load.py semantic   # 按语义（段落合并）切

怎么算成功：
  看到 "共切出 N 个片段"，且 N 在 200~2000 之间（太少说明资料不够，太多说明没切好）
"""
import os
import re
import sys
import json

import pdfplumber
from docx import Document

DATA_DIR = "data"
OUT_FILE = "chunks.json"


# ---------------- 1. 读文件：把不同格式都变成纯文本 ----------------
def read_pdf(path):
    """读 PDF。pdfplumber 逐页提取文字。"""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_docx(path):
    """读 Word。逐段落提取文字。"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(path):
    """读 txt。网页另存的 .txt 或手工整理的资料。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_any(path):
    low = path.lower()
    if low.endswith(".pdf"):
        return read_pdf(path)
    if low.endswith(".docx"):
        return read_docx(path)
    if low.endswith(".txt") or low.endswith(".md"):
        return read_txt(path)
    return ""


# ---------------- 2. 切分：三种策略，后面要对比效果 ----------------
def split_fixed(text, size=300, overlap=50):
    """
    策略 A：固定长度切分（最基础，作为实验对照组）
    每 300 字一块，相邻块重叠 50 字（避免一句话被截断在两个块里）
    """
    text = re.sub(r"\n{2,}", "\n", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if len(piece) >= 30:
            chunks.append(piece)
        i += size - overlap
    return chunks


def split_section(text):
    """
    策略 B：按章节/条款切分
    遇到 "第一章" "第三条" "附件1" "1." 这类标题就开新块
    """
    lines = text.split("\n")
    chunks, cur = [], ""
    title_re = re.compile(r"^\s*(第[一二三四五六七八九十百\d]+[章节条部分]|[（(][一二三四五六七八九十\d]+[)）]|附件\s*\d*|\d+[\.、]\s*\S)")
    for ln in lines:
        if title_re.match(ln) and len(cur.strip()) > 30:
            chunks.append(cur.strip())
            cur = ln
        else:
            cur += "\n" + ln
    if len(cur.strip()) > 30:
        chunks.append(cur.strip())
    return chunks


def split_semantic(text, max_size=400, min_size=120):
    """
    策略 C：语义切分（按段落合并，凑到 120~400 字就成块）
    效果通常最好：块内语义完整，不会被从中间切断
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    paras = [re.sub(r"\s+", " ", p) for p in paras]
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + len(p) <= max_size:
            cur = cur + "\n" + p
        else:
            if len(cur) >= min_size:
                chunks.append(cur)
                cur = p
            else:
                cur = cur + "\n" + p
    if len(cur.strip()) >= 30:
        chunks.append(cur.strip())
    return chunks


# ---------------- 3. 主流程 ----------------
SPLITTERS = {
    "fixed": split_fixed,
    "section": split_section,
    "semantic": split_semantic,
}


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    if strategy not in SPLITTERS:
        print("可选策略：fixed / section / semantic")
        return

    if not os.path.isdir(DATA_DIR):
        print(f"没找到 {DATA_DIR}/ 目录，请先建一个并把资料放进去")
        return

    files = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]
    if not files:
        print(f"{DATA_DIR}/ 是空的，请先把 PDF/Word/txt 资料放进去")
        return

    splitter = SPLITTERS[strategy]
    chunks = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        try:
            text = read_any(path)
        except Exception as e:
            print(f"  [跳过] {fname} 读取失败：{e}")
            continue
        if len(text.strip()) < 50:
            print(f"  [跳过] {fname} 内容太少（可能是扫描版 PDF，没有文字层）")
            continue
        pieces = splitter(text)
        for p in pieces:
            chunks.append({"text": p, "source": fname})
        print(f"  已读 {fname} -> {len(pieces)} 块")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"strategy": strategy, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"策略：{strategy}　共切出 {len(chunks)} 个片段，已写入 {OUT_FILE}")
    if chunks:
        lens = [len(c["text"]) for c in chunks]
        print(f"片段平均长度 {sum(lens)//len(lens)} 字，最短 {min(lens)}，最长 {max(lens)}")


if __name__ == "__main__":
    main()
