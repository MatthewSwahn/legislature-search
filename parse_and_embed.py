from lxml import etree
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os
from datetime import datetime
sys.path.append(os.getcwd())

import parser

def generate_sentence_embeddings(texts, model, batch_size=32):
    """
    Generate sentence embeddings for a list of texts using sentence-transformers.
    
    Args:
        texts: List of text strings
        model: Sentence transformer model
        batch_size: Number of texts to process at once
    
    Returns:
        numpy array of sentence embeddings
    """
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/processed-data/sentence_embeddings_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
    # Parse XML data
    root = etree.parse('data/BILLS-119hr1enr.xml')
    parsed_data = parser.parse_xml(root=root)
    
    df = pd.DataFrame(parsed_data)
    
    # remove empty text
    print('Number of empty text values:', sum(df['text'].str.strip() != ''))
    df = df[df['text'].str.strip() != '']
    
    # Generate embeddings
    texts = df['text'].tolist()
    embeddings = generate_sentence_embeddings(texts, model)
    
    df['embedding'] = embeddings.tolist()

    # Save embeddings separately as numpy array for efficiency
    np.save(f'{output_dir}/sentence_embeddings.npy', embeddings)
    
    # Save a sample with embeddings for inspection
    sample_df = df.head(10).copy()
    sample_df.to_json(f'{output_dir}/sample_with_embeddings.json', orient='records', indent=2)
    
    print("Files saved:")
    print(f"- {output_dir}/sentence_embeddings.npy (embeddings array)")
    print(f"- {output_dir}/sample_with_embeddings.json (sample with embeddings)")
    
    print(f"\nProcessing complete! Generated embeddings for {len(df)} text segments.")

if __name__ == "__main__":
    main()