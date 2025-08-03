from lxml import etree
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch
import sys
import os
from datetime import datetime
sys.path.append(os.getcwd())

import parser

def generate_multilayer_roberta_embeddings(texts, model, tokenizer, batch_size=32):
    """
    Generate RoBERTa embeddings from multiple layers.
    
    Args:
        texts: List of text strings
        model: RoBERTa model
        tokenizer: RoBERTa tokenizer
        batch_size: Number of texts to process at once
    
    Returns:
        Dictionary with embeddings from different layers and pooling strategies
    """
    # Store embeddings from different strategies
    embedding_strategies = {
        'last_layer_cls': [],           # [CLS] from final layer (our current approach)
        'last_layer_mean': [],          # Mean pooling from final layer
        'second_last_cls': [],          # [CLS] from second-to-last layer  
        'last_4_layers_cls': [],        # [CLS] concatenated from last 4 layers
        'last_4_layers_mean': [],       # Mean of [CLS] from last 4 layers
        'all_layers_weighted': []       # Weighted average of all layers
    }
    
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
        
        # Generate embeddings with all hidden states
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            
            # Get all hidden states (13 layers for RoBERTa-base: embedding + 12 transformer layers)
            hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
            attention_mask = encoded['attention_mask']
            
            batch_size_actual = hidden_states[0].shape[0]
            
            for batch_idx in range(batch_size_actual):
                # Strategy 1: [CLS] from final layer (layer -1)
                cls_final = hidden_states[-1][batch_idx, 0, :].numpy()
                embedding_strategies['last_layer_cls'].append(cls_final)
                
                # Strategy 2: Mean pooling from final layer
                mask = attention_mask[batch_idx].unsqueeze(-1).expand_as(hidden_states[-1][batch_idx])
                masked_embeddings = hidden_states[-1][batch_idx] * mask
                mean_final = masked_embeddings.sum(dim=0) / mask.sum(dim=0)
                embedding_strategies['last_layer_mean'].append(mean_final.numpy())
                
                # Strategy 3: [CLS] from second-to-last layer (layer -2)
                cls_second_last = hidden_states[-2][batch_idx, 0, :].numpy()
                embedding_strategies['second_last_cls'].append(cls_second_last)
                
                # Strategy 4: [CLS] concatenated from last 4 layers
                last_4_cls = torch.cat([
                    hidden_states[-4][batch_idx, 0, :],
                    hidden_states[-3][batch_idx, 0, :], 
                    hidden_states[-2][batch_idx, 0, :],
                    hidden_states[-1][batch_idx, 0, :]
                ], dim=0)
                embedding_strategies['last_4_layers_cls'].append(last_4_cls.numpy())
                
                # Strategy 5: Mean of [CLS] from last 4 layers
                last_4_cls_mean = torch.stack([
                    hidden_states[-4][batch_idx, 0, :],
                    hidden_states[-3][batch_idx, 0, :],
                    hidden_states[-2][batch_idx, 0, :], 
                    hidden_states[-1][batch_idx, 0, :]
                ]).mean(dim=0)
                embedding_strategies['last_4_layers_mean'].append(last_4_cls_mean.numpy())
                
                # Strategy 6: Weighted average of all layers (give more weight to later layers)
                weights = torch.softmax(torch.arange(len(hidden_states), dtype=torch.float), dim=0)
                weighted_cls = torch.zeros_like(hidden_states[0][batch_idx, 0, :])
                for layer_idx, hidden_state in enumerate(hidden_states):
                    weighted_cls += weights[layer_idx] * hidden_state[batch_idx, 0, :]
                embedding_strategies['all_layers_weighted'].append(weighted_cls.numpy())
    
    # Convert lists to numpy arrays
    for strategy in embedding_strategies:
        embedding_strategies[strategy] = np.vstack(embedding_strategies[strategy])
    
    return embedding_strategies

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/processed-data/multilayer_embeddings_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()
    
    # Parse XML data
    root = etree.parse('data/BILLS-119hr1enr.xml')
    parsed_data = parser.parse_xml(root=root)
    
    df = pd.DataFrame(parsed_data)
    
    # Filter empty text
    df = df[df['text'].str.strip() != '']
    
    # Generate embeddings
    texts = df['text'].tolist()
    embedding_strategies = generate_multilayer_roberta_embeddings(texts, model, tokenizer)
    
    # Save embeddings
    for strategy, embeddings in embedding_strategies.items():
        np.save(f'{output_dir}/embeddings_{strategy}.npy', embeddings)
    
    # Save sample for inspection
    sample_df = df.head(10).copy()
    sample_df.to_json(f'{output_dir}/sample.json', orient='records', indent=2)
    
    print(f"Processing complete! Generated {len(embedding_strategies)} embedding strategies for {len(df)} segments.")

if __name__ == "__main__":
    main()