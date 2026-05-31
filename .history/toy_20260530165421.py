import os
import sys
from tex2docx import LatexToWordConverter

# Ensure pandoc & pandoc-crossref are found (Git Bash PATH caching workaround)
_PANDOC_DIR = "D:/pandoc/pandoc-3.9.0.2"
_CROSSREF_DIR = "D:/pandoc/pandoc-crossref-Windows-X64"
os.environ["PATH"] = _PANDOC_DIR + os.pathsep + _CROSSREF_DIR + os.pathsep + os.environ.get("PATH", "")

config = {
    'input_texfile': 'article/main_paper.tex',
    'output_docxfile': 'formula-ref.docx',
    'reference_docfile': 'article/references.bib',
    'cslfile': 'article/gbt7714-numeric.csl',
    'bibfile': 'article/references.bib',
    'fix_table': True,
    'debug': True
}

if __name__ == '__main__':
    converter = LatexToWordConverter(**config)
    converter.convert()
