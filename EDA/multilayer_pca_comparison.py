import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import re
import os
from pathlib import Path

def extract_title_id(xpath):
    """Extract title ID from xpath"""
    if '/title[' in xpath:
        match = re.search(r'/title\[(\d+)\]', xpath)
        if match:
            title_num = int(match.group(1))
            roman_numerals = {
                1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 
                8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"
            }
            return f"Title {roman_numerals.get(title_num, str(title_num))}"
    else:
        return "Table of Contents"

def find_latest_multilayer_dir():
    """Find the most recent multilayer embeddings directory"""
    processed_dir = Path("data/processed-data")
    if not processed_dir.exists():
        raise FileNotFoundError("data/processed-data directory not found")
    
    embedding_dirs = list(processed_dir.glob("multilayer_embeddings_*"))
    if not embedding_dirs:
        raise FileNotFoundError("No multilayer embeddings directories found")
    
    return max(embedding_dirs)

def compare_embedding_strategies():
    """Compare PCA results across different embedding strategies"""
    
    # Find the latest multilayer embeddings directory
    embeddings_dir = find_latest_multilayer_dir()
    print(f"Using embeddings from: {embeddings_dir}")
    
    # Load metadata
    metadata_df = pd.read_json(embeddings_dir / "sample.json")
    metadata_df['title_id'] = metadata_df['xpath'].apply(extract_title_id)
    
    # Get all embedding strategy files
    embedding_files = list(embeddings_dir.glob("embeddings_*.npy"))
    strategies = [f.stem.replace('embeddings_', '') for f in embedding_files]
    
    print(f"Found {len(strategies)} embedding strategies")
    
    # Create subplots for comparison
    n_strategies = len(strategies)
    n_cols = 3
    n_rows = (n_strategies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    # Get unique titles for consistent coloring
    unique_titles = sorted(metadata_df['title_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_titles)))
    
    pca_results = {}
    
    for i, strategy in enumerate(strategies):
        # Load embeddings
        embeddings = np.load(embeddings_dir / f"embeddings_{strategy}.npy")
        
        # Perform PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Store results
        pca_results[strategy] = {
            'embeddings_2d': embeddings_2d,
            'explained_variance': pca.explained_variance_ratio_,
            'total_variance': pca.explained_variance_ratio_.sum()
        }
        
        # Plot
        ax = axes[i]
        for j, title in enumerate(unique_titles):
            title_mask = metadata_df['title_id'] == title
            if title_mask.any():
                ax.scatter(embeddings_2d[title_mask, 0], embeddings_2d[title_mask, 1], 
                          c=[colors[j]], label=title, alpha=0.7, s=30)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax.set_title(f'{strategy.replace("_", " ").title()}\nTotal Variance: {pca.explained_variance_ratio_.sum():.3f}')
        
        if i == 0:  # Only show legend for first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_strategies, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the comparison plot
    output_path = "EDA/multilayer_pca_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nPCA Variance Explained Summary:")
    print("-" * 50)
    for strategy, results in pca_results.items():
        print(f"{strategy.replace('_', ' ').title():25}: {results['total_variance']:.4f}")
    
    # Find best strategy
    best_strategy = max(pca_results.keys(), key=lambda k: pca_results[k]['total_variance'])
    print(f"\nBest strategy (highest variance explained): {best_strategy.replace('_', ' ').title()}")
    
    return pca_results

def main():
    try:
        results = compare_embedding_strategies()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run parse_and_embed_multilayer.py first to generate embeddings.")

if __name__ == "__main__":
    main()