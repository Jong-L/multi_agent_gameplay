from tex2docx import LatexToWordConverter

config = {
    'input_texfile': 'article/main_paper.tex',
    'output_docxfile': 'formula-ref.docx',
    'reference_docfile': 'article/references.bib',
    'cslfile': 'article\gbt7714-numeric.csl',
    'bibfile': 'article/references.bib',
    'fix_table': True,
    'debug': False
}

converter = LatexToWordConverter(**config)
converter.convert()