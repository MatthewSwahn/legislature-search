from lxml import etree
import pandas as pd
from transformers import BigBirdTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

import parser

# v2 was on 119hr1eh, v3 is 119hr1enr.html (the final bill) https://www.govinfo.gov/app/details/BILLS-119hr1enr/summary

# import data
root = etree.parse('data/congress_119_hr1/BILLS-119hr1enr.xml')

bbb_data = parser.parse_xml(root=root)

bbb_df = pd.DataFrame(bbb_data)

print(bbb_df.info())
print(bbb_df.head())


# instantiate tokenizer. Leaning bigbird as the number of tokens is large
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-large')
#print('tok enizer max len', tokenizer.model_max_length) # 4096

# tokenize text
encoded = tokenizer(bbb_df.text.to_list(), padding=True, truncation=True, return_tensors='pt')

input_ids_list = encoded['input_ids'].tolist()

# get number of tokens as column
pad_token_id = tokenizer.pad_token_id
bbb_df['num_spm_tokens_by_subpara'] = [
    sum(token_id != pad_token_id for token_id in input_ids)
    for input_ids in input_ids_list
]
# how much text goes to the 4096 token limit
bbb_df['max_token_ind'] = bbb_df['num_spm_tokens_by_subpara'].apply(lambda x: 1 if x == 4096 else 0)

print('how much text hits 4096 tokens?', bbb_df['max_token_ind'].value_counts()) # 7

print(bbb_df['xpath'][bbb_df['max_token_ind'] ==1])

bbb_df[bbb_df['max_token_ind'] ==1].to_json('why.json')