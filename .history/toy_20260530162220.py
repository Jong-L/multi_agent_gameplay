from tex2docx import LatexToWordConverter

config = {
    'input_texfile': '<your_texfile>',
    'output_docxfile': '<your_docxfile>',
    'reference_docfile': '<your_reference_docfile>',
    'cslfile': '<your_cslfile>',
    'bibfile': '<your_bibfile>',
    'fix_table': True,
    'debug': False
}

converter = LatexToWordConverter(**config)
converter.convert()