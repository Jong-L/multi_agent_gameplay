#!/usr/bin/env python3
"""统计 LaTeX 正文中文字数（含数学公式内汉字，不含纯英文/数字/标点）"""

import re

TEX_FILE = r"D:\schoolTour\softwares\multi-agent-gameplay\article\main_paper.tex"

def extract_body(text):
    """提取 \begin{document} ... \end{document} 之间的内容"""
    m = re.search(r'\\begin\{document\}', text)
    if not m:
        return text
    start = m.end()
    m2 = re.search(r'\\end\{document\}', text[start:])
    end = start + m2.start() if m2 else len(text)
    return text[start:end]

def strip_comments(text):
    """移除 % 注释（注意不要误删 \%）"""
    lines = []
    for line in text.splitlines():
        # 只去掉行首空白后第一个 %（非反斜杠转义）之后的内容
        stripped = re.sub(r'(?<!\\)%.*', '', line)
        lines.append(stripped)
    return '\n'.join(lines)

def is_cjk_char(ch):
    """判断是否为中日韩字符（含标点）"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF   # 常用汉字
        or 0x3400 <= code <= 0x4DBF  # 扩展A
        or 0xF900 <= code <= 0xFAFF  # 兼容汉字
        or 0x3000 <= code <= 0x303F  # 中文标点
        or 0xFF00 <= code <= 0xFFEF  # 全角ASCII
    )

def count_chinese_chars(text):
    """统计中文字符（含标点）数量"""
    return sum(1 for ch in text if is_cjk_char(ch))

def count_chinese_words(text):
    """
    统计"中文字数"（学术惯例）：
    连续 CJK 字符序列算 1 个词；每个 CJK 标点单独计数。
    非 CJK 内容（英文/数学/数字）不计入。
    """
    # 先把非 CJK 区域替换为空格，再按空白/连续 CJK 分词
    cleaned = ''.join(ch if is_cjk_char(ch) else ' ' for ch in text)
    tokens = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+', cleaned)
    return len(tokens)

def main():
    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        raw = f.read()

    body = extract_body(raw)
    no_comments = strip_comments(body)

    # 去掉 LaTeX 命令（\cmd{...}、\cmd[...]{...}、\cmd）
    no_cmds = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{[^}]*\}', '', no_comments)
    no_cmds = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', '', no_cmds)

    # 去掉数学环境（$...$、$$...$$、\[...\]、\(...\)）
    no_math = re.sub(r'\$\$.*?\$\$', ' ', no_cmds, flags=re.DOTALL)
    no_math = re.sub(r'\$.*?\$', ' ', no_math, flags=re.DOTALL)
    no_math = re.sub(r'\\\[.*?\\\]', ' ', no_math, flags=re.DOTALL)
    no_math = re.sub(r'\\\(.*?\\\)', ' ', no_math, flags=re.DOTALL)

    char_count = count_chinese_chars(no_math)
    word_count = count_chinese_words(no_math)

    print(f"文件: {TEX_FILE}")
    print(f"中文标点+汉字字符数: {char_count}")
    print(f"中文字数（连续CJK序列计1词）: {word_count}")
    print(f"说明：字符数含标点；词数以连续CJK块为单位")

if __name__ == '__main__':
    main()
