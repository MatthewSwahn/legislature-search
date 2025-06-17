from lxml import etree
import pandas as pd
from transformers import BigBirdTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

import text_by_part as tbp


# import data
root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

bbb_data = tbp.get_text_from_xml(root=root)

bbb_df = pd.DataFrame(bbb_data)

# instantiate tokenizer. Leaning bigbird as the number of tokens is large
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-large')
print('tokenizer max len', tokenizer.model_max_length)
# tokenize text
encoded = tokenizer(bbb_df.text.to_list(), padding=True, truncation=True, return_tensors='pt')

input_ids_list = encoded['input_ids'].tolist()

# get number of tokens as column
bbb_df['num_spm_tokens'] = [len(x) for x in input_ids_list]

sns.histplot(data=bbb_df, x="num_spm_tokens", kde=True, bins=30)
plt.savefig('EDA/spm-token-dist.png', dpi=300)

# current xml parsing gets text on the section level and all hit 4096 tokens