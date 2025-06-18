from lxml import etree
import pandas as pd
from transformers import BigBirdTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

import parser


# import data
root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

bbb_data = parser.get_text_subparagraph_level(root=root)

bbb_df = pd.DataFrame(bbb_data)


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

bbb_df.to_csv('EDA/bbb_df_by_subpara_v1.csv')
bbb_df[bbb_df['max_token_ind'] == 1].to_csv('EDA/bbb_df_by_subpara_max_tokens.csv')

sns.histplot(data=bbb_df, x="num_spm_tokens_by_subpara", kde=True, bins=30)
plt.savefig('EDA/spm_token_dist_by_subpara_v1.png', dpi=300)
