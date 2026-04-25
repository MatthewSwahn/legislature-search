#!/usr/bin/env python3
"""
XML to Embeddings Pipeline

Parses XML files, generates embeddings using various strategies, and saves as Parquet format.
Based on parser.py and parse_and_embed_multilayer.py functionality.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from lxml import etree
import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer

# Add current directory to path to import parser module
sys.path.append(os.getcwd())
import parser

def generate_sentence_transformer_embedding(texts, batch_size=32):
    """
    Generate embeddings using SentenceTransformers with all-MiniLM-L6-v2 model.
    
    Args:
        texts: List of text strings
        batch_size: Number of texts to process at once
    
    Returns:
        numpy array of embeddings
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings)

def generate_roberta_embedding(texts, model, tokenizer, strategy, batch_size=32):
    """
    Generate RoBERTa embeddings using a specified strategy.
    
    Args:
        texts: List of text strings
        model: RoBERTa model
        tokenizer: RoBERTa tokenizer  
        strategy: Embedding strategy name
        batch_size: Number of texts to process at once
    
    Returns:
        numpy array of embeddings
    """
    embeddings = []
    
    # Process in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate embeddings with all hidden states for multi-layer strategies
        with torch.no_grad():
            if strategy in ['second_last_cls', 'last_4_layers_cls', 'last_4_layers_mean', 'all_layers_weighted']:
                outputs = model(**encoded, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            else:
                outputs = model(**encoded)
                hidden_states = None
            
            attention_mask = encoded['attention_mask']
            batch_size_actual = encoded['input_ids'].shape[0]
            
            for batch_idx in range(batch_size_actual):
                if strategy == 'last_layer_cls':
                    # [CLS] from final layer (default RoBERTa approach)
                    embedding = outputs.last_hidden_state[batch_idx, 0, :].numpy()
                    
                elif strategy == 'last_layer_mean':
                    # Mean pooling from final layer
                    mask = attention_mask[batch_idx].unsqueeze(-1).expand_as(outputs.last_hidden_state[batch_idx])
                    masked_embeddings = outputs.last_hidden_state[batch_idx] * mask
                    embedding = (masked_embeddings.sum(dim=0) / mask.sum(dim=0)).numpy()
                    
                elif strategy == 'second_last_cls':
                    # [CLS] from second-to-last layer
                    embedding = hidden_states[-2][batch_idx, 0, :].numpy()
                    
                elif strategy == 'last_4_layers_cls':
                    # [CLS] concatenated from last 4 layers
                    last_4_cls = torch.cat([
                        hidden_states[-4][batch_idx, 0, :],
                        hidden_states[-3][batch_idx, 0, :], 
                        hidden_states[-2][batch_idx, 0, :],
                        hidden_states[-1][batch_idx, 0, :]
                    ], dim=0)
                    embedding = last_4_cls.numpy()
                    
                elif strategy == 'last_4_layers_mean':
                    # Mean of [CLS] from last 4 layers
                    last_4_cls_mean = torch.stack([
                        hidden_states[-4][batch_idx, 0, :],
                        hidden_states[-3][batch_idx, 0, :],
                        hidden_states[-2][batch_idx, 0, :], 
                        hidden_states[-1][batch_idx, 0, :]
                    ]).mean(dim=0)
                    embedding = last_4_cls_mean.numpy()
                    
                elif strategy == 'all_layers_weighted':
                    # Weighted average of all layers (give more weight to later layers)
                    weights = torch.softmax(torch.arange(len(hidden_states), dtype=torch.float), dim=0)
                    weighted_cls = torch.zeros_like(hidden_states[0][batch_idx, 0, :])
                    for layer_idx, hidden_state in enumerate(hidden_states):
                        weighted_cls += weights[layer_idx] * hidden_state[batch_idx, 0, :]
                    embedding = weighted_cls.numpy()
                    
                else:
                    raise ValueError(f"Unknown RoBERTa embedding strategy: {strategy}")
                
                embeddings.append(embedding)
    
    return np.vstack(embeddings)

def main():
    parser_args = argparse.ArgumentParser(description='Parse XML and create embeddings saved as Parquet')
    parser_args.add_argument('xml_file', help='Path to XML file to parse')
    parser_args.add_argument('strategy', choices=[
        'sentence_transformer',
        'last_layer_cls', 
        'last_layer_mean', 
        'second_last_cls', 
        'last_4_layers_cls', 
        'last_4_layers_mean', 
        'all_layers_weighted'
    ], help='Embedding strategy to use')
    parser_args.add_argument('--output-dir', default='data', help='Output directory (default: data)')
    parser_args.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding generation (default: 32)')
    
    args = parser_args.parse_args()
    
    # Validate XML file exists
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = xml_path.stem
    output_file = output_dir / f"{filename_base}_{args.strategy}_{timestamp}.parquet"
    
    print(f"Processing XML file: {xml_path}")
    print(f"Using embedding strategy: {args.strategy}")
    print(f"Output file: {output_file}")
    
    try:
        # Parse XML data first
        print("Parsing XML...")
        root = etree.parse(str(xml_path))
        parsed_data = parser.parse_xml(root=root)
        
        # Create dataframe
        df = pd.DataFrame(parsed_data)
        
        # Filter out empty text
        original_count = len(df)
        df = df[df['text'].str.strip() != '']
        filtered_count = len(df)
        
        print(f"Parsed {original_count} segments, kept {filtered_count} non-empty segments")
        
        if filtered_count == 0:
            print("Error: No valid text segments found in XML")
            sys.exit(1)
        
        texts = df['text'].tolist()
        
        # Generate embeddings based on strategy
        if args.strategy == 'sentence_transformer':
            print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            embeddings = generate_sentence_transformer_embedding(texts, args.batch_size)
        else:
            print("Loading RoBERTa model and tokenizer...")
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')
            model.eval()
            
            print(f"Generating RoBERTa embeddings using {args.strategy} strategy...")
            embeddings = generate_roberta_embedding(texts, model, tokenizer, args.strategy, args.batch_size)
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Add embeddings to dataframe
        # Convert numpy array to list of lists for parquet compatibility
        df['embedding'] = embeddings.tolist()
        
        # Add metadata columns
        df['embedding_strategy'] = args.strategy
        df['embedding_dim'] = embeddings.shape[1]
        df['timestamp'] = timestamp
        df['source_file'] = str(xml_path)
        
        # Save as parquet
        print(f"Saving to parquet format...")
        df.to_parquet(output_file, index=False)
        
        print(f"Successfully saved {len(df)} segments with embeddings to: {output_file}")
        print(f"Embedding dimensions: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()