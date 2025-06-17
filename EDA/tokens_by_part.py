import text_by_part as tbp
from lxml import etree
import pandas as pd
import sentencepiece as spm

root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

parts = tbp.get_text_from_xml(root=root)

parts_pd = pd.DataFrame(parts)
print(parts_pd.info())