from text_by_part import get_text_from_xml
from lxml import etree
import pandas as pd

root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

parts = get_text_from_xml(root=root)

parts_pd = pd.DataFrame(parts)
parts_pd['text_len'] = parts_pd.text.apply(len)
#print(parts_pd.info()) #2226 rows
print(parts_pd.head())


max_text_len = max(parts_pd['text_len'])
longest_str = parts_pd['text'][parts_pd['text_len']==max_text_len]

#print(longest_str.values)