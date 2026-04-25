import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import re
import os
from pathlib import Path

def extract_title_id(xpath):
    """
    Extract title ID from xpath.
    Examples:
    - /legis-body/title[1]/... -> "Title I"
    - /legis-body/title[7]/... -> "Title VII"  
    - /legis-body/section -> "Table of Contents"
    """
    if '/title[' in xpath:
        # Extract title number from xpath
        match = re.search(r'/title\[(\d+)\]', xpath)
        if match:
            title_num = int(match.group(1))
            # Convert to Roman numerals
            roman_numerals = {
                1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 
                8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"
            }
            return f"Title {roman_numerals.get(title_num, str(title_num))}"
    else:
        # Handle cases without title (like table of contents)
        return "Table of Contents"

def find_latest_embeddings_dir():
    """Find the most recent embeddings directory"""
    processed_dir = Path("data/processed-data")
    if not processed_dir.exists():
        raise FileNotFoundError("data/processed-data directory not found")
    
    # Find all roberta embedding directories
    embedding_dirs = list(processed_dir.glob("roberta_embeddings_*"))
    if not embedding_dirs:
        raise FileNotFoundError("No RoBERTa embeddings directories found")
    
    # Return the most recent one (by name which includes timestamp)
    return max(embedding_dirs)

def main():
    print("Loading RoBERTa embeddings data...")
    
    # Find the latest embeddings directory
    embeddings_dir = find_latest_embeddings_dir()
    print(f"Using embeddings from: {embeddings_dir}")
    
    # Load embeddings and metadata
    embeddings = np.load(embeddings_dir / "roberta_embeddings.npy")
    metadata_df = pd.read_csv(embeddings_dir / "parsed_text_metadata.csv")
    
    print(f"Loaded {len(metadata_df)} text segments with {embeddings.shape[1]}-dimensional embeddings")
    
    # Extract title information
    metadata_df['title_id'] = metadata_df['xpath'].apply(extract_title_id)
    
    # Print title distribution
    print("\nTitle distribution:")
    title_counts = metadata_df['title_id'].value_counts().sort_index()
    print(title_counts)
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Add PCA components to dataframe
    metadata_df['pca_1'] = embeddings_2d[:, 0]
    metadata_df['pca_2'] = embeddings_2d[:, 1]
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Get unique titles for consistent coloring
    unique_titles = sorted(metadata_df['title_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_titles)))
    
    # Create the scatter plot
    for i, title in enumerate(unique_titles):
        title_data = metadata_df[metadata_df['title_id'] == title]
        plt.scatter(title_data['pca_1'], title_data['pca_2'], 
                   c=[colors[i]], label=title, alpha=0.7, s=50)
    
    plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA Visualization of RoBERTa Embeddings by Title')
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    # Save the plot
    output_path = "EDA/bill_pca_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Save PCA results for further analysis
    results_df = metadata_df[['xpath', 'tag', 'title_id', 'text_length', 'word_count', 'token_count', 'pca_1', 'pca_2']].copy()
    results_path = embeddings_dir / "pca_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"PCA results saved to: {results_path}")
    
    # Print some statistics
    print(f"\nPCA Statistics:")
    print(f"PC1 range: [{embeddings_2d[:, 0].min():.3f}, {embeddings_2d[:, 0].max():.3f}]")
    print(f"PC2 range: [{embeddings_2d[:, 1].min():.3f}, {embeddings_2d[:, 1].max():.3f}]")
    
    # Show which titles are most separated
    print(f"\nTitle centroids in PCA space:")
    for title in unique_titles:
        title_data = metadata_df[metadata_df['title_id'] == title]
        centroid_1 = title_data['pca_1'].mean()
        centroid_2 = title_data['pca_2'].mean()
        print(f"{title}: PC1={centroid_1:.3f}, PC2={centroid_2:.3f} (n={len(title_data)})")

if __name__ == "__main__":
    main()