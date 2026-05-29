import re

with open('D:/schoolTour/softwares/multi-agent-gameplay/article/main_paper.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove comment lines and inline comments
lines = content.split('\n')
cleaned = []
for line in lines:
    if line.strip().startswith('%'):
        continue
    idx = line.find('%')
    if idx >= 0:
        line = line[:idx]
    cleaned.append(line)

text = '\n'.join(cleaned)

# Extract document body
doc_match = re.search(r'\\begin\{document\}.*?\\end\{document\}', text, re.DOTALL)
if doc_match:
    text = doc_match.group()

# Remove LaTeX commands and their arguments
text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})+', '', text)
# Remove remaining math
text = re.sub(r'\$[^$]*\$', '', text)
# Remove display math brackets
text = re.sub(r'\\\[', '', text)
text = re.sub(r'\\\]', '', text)
text = re.sub(r'\\\(', '', text)
text = re.sub(r'\\\)', '', text)

chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
english_words = len(re.findall(r'[a-zA-Z]+', text))

total = chinese_chars + english_words
print('中文字符数:', chinese_chars)
print('英文单词数:', english_words)
print('正文字数(中文+英文):', total)
