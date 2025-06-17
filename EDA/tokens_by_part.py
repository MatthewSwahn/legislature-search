from Legislature-Search.parsing.text_by_part import get_text_from_xml
from lxml import etree
import pandas as pd
import sentencepiece as spm

root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

parts = get_text_from_xml(root=root)

parts_pd = pd.DataFrame(parts)
print(parts_pd.info())Legi