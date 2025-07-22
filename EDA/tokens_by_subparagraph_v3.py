from lxml import etree
import pandas as pd
from transformers import BigBirdTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

import parser2

# v2 was on 119hr1eh, v3 is 119hr1enr.html (the final bill) https://www.govinfo.gov/app/details/BILLS-119hr1enr/summary

# import data
root = etree.parse('data/congress_119_hr1/BILLS-119hr1enr.xml')

bbb_data = parser2.parse_xml(root=root)

bbb_df = pd.DataFrame(bbb_data)

print(bbb_df.info())
print(bbb_df.head())