from lxml import etree
import pandas as pd
from transformers import RobertaTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

import parser

# v4 uses RoBERTa tokenizer instead of BigBird

# import data
root = etree.parse('data/BILLS-119hr1enr.xml')

bbb_data = parser.parse_xml(root=root)

bbb_df = pd.DataFrame(bbb_data)

print(bbb_df.info())
print(bbb_df.head())


# instantiate RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print('RoBERTa tokenizer max len', tokenizer.model_max_length) # 512

# tokenize text
encoded = tokenizer(bbb_df.text.to_list(), padding=True, truncation=True, return_tensors='pt')

input_ids_list = encoded['input_ids'].tolist()

# get number of tokens as column
pad_token_id = tokenizer.pad_token_id
bbb_df['num_roberta_tokens_by_subpara'] = [
    sum(token_id != pad_token_id for token_id in input_ids)
    for input_ids in input_ids_list
]
# how much text goes to the 512 token limit (RoBERTa's max length)
bbb_df['max_token_ind'] = bbb_df['num_roberta_tokens_by_subpara'].apply(lambda x: 1 if x == 512 else 0)

print('how much text hits 512 tokens?', bbb_df['max_token_ind'].value_counts())

print(bbb_df['xpath'][bbb_df['max_token_ind'] == 1])

bbb_df[bbb_df['max_token_ind'] == 1].to_json('roberta_why.json')