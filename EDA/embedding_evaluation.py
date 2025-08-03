import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import pdist, squareform
import re
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

def find_metadata_file(npy_path):
    """Find the corresponding metadata file for the given npy file"""
    npy_path = Path(npy_path)
    
    # Look for sample_with_embeddings.json first, then sample.json
    json_files = ["sample_with_embeddings.json", "sample.json"]
    
    for json_filename in json_files:
        json_file = npy_path.parent / json_filename
        if json_file.exists():
            return json_file
    
    raise FileNotFoundError(f"Could not find metadata file (sample_with_embeddings.json or sample.json) in {npy_path.parent}")

def calculate_clustering_metrics(embeddings, labels):
    """Calculate clustering quality metrics"""
    metrics = {}
    
    # Silhouette Score (higher is better, range [-1, 1])
    metrics['silhouette_score'] = silhouette_score(embeddings, labels)
    
    # Calinski-Harabasz Index (higher is better)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
    
    # Davies-Bouldin Index (lower is better)
    metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
    
    return metrics

def calculate_classification_metrics(embeddings, labels):
    """Calculate classification-based metrics"""
    metrics = {}
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # k-NN Classification (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, embeddings, labels_encoded, cv=5, scoring='accuracy')
    metrics['knn_accuracy_mean'] = knn_scores.mean()
    metrics['knn_accuracy_std'] = knn_scores.std()
    
    # Logistic Regression (Linear Separability)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, embeddings, labels_encoded, cv=5, scoring='accuracy')
    metrics['linear_separability_mean'] = lr_scores.mean()
    metrics['linear_separability_std'] = lr_scores.std()
    
    return metrics

def calculate_distance_metrics(embeddings, labels):
    """Calculate within-group vs between-group distance metrics"""
    metrics = {}
    
    # Calculate cosine similarity matrix
    cos_sim_matrix = cosine_similarity(embeddings)
    
    # Calculate within-group and between-group similarities
    unique_labels = np.unique(labels)
    within_group_sims = []
    between_group_sims = []
    
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i < j:  # Avoid double counting
                sim = cos_sim_matrix[i, j]
                if label_i == label_j:
                    within_group_sims.append(sim)
                else:
                    between_group_sims.append(sim)
    
    metrics['within_group_cosine_mean'] = np.mean(within_group_sims)
    metrics['between_group_cosine_mean'] = np.mean(between_group_sims)
    metrics['within_between_ratio'] = np.mean(within_group_sims) / np.mean(between_group_sims)
    
    # Nearest Neighbor Purity (k=5)
    k = 5
    purity_scores = []
    
    for i in range(len(embeddings)):
        # Find k nearest neighbors (excluding self)
        similarities = cos_sim_matrix[i]
        nearest_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self (index i)
        
        # Calculate purity (fraction of neighbors with same label)
        same_label_count = sum(labels[idx] == labels[i] for idx in nearest_indices)
        purity_scores.append(same_label_count / k)
    
    metrics['nearest_neighbor_purity'] = np.mean(purity_scores)
    
    return metrics

def calculate_title_centroid_distances(embeddings, labels):
    """Calculate distances between title centroids"""
    unique_labels = np.unique(labels)
    centroids = {}
    
    # Calculate centroids for each title
    for label in unique_labels:
        mask = labels == label
        centroids[label] = np.mean(embeddings[mask], axis=0)
    
    # Calculate pairwise distances between centroids
    centroid_distances = {}
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
                centroid_distances[f"{label_i}_vs_{label_j}"] = dist
    
    return centroid_distances, centroids

def evaluate_embedding_strategy(embeddings, labels, strategy_name):
    """Comprehensive evaluation of a single embedding strategy"""
    print(f"\nEvaluating {strategy_name}...")
    
    results = {'strategy': strategy_name}
    
    # Clustering metrics
    clustering_metrics = calculate_clustering_metrics(embeddings, labels)
    results.update(clustering_metrics)
    
    # Classification metrics
    classification_metrics = calculate_classification_metrics(embeddings, labels)
    results.update(classification_metrics)
    
    # Distance metrics
    distance_metrics = calculate_distance_metrics(embeddings, labels)
    results.update(distance_metrics)
    
    # Centroid distances
    centroid_distances, centroids = calculate_title_centroid_distances(embeddings, labels)
    results['avg_centroid_distance'] = np.mean(list(centroid_distances.values()))
    
    return results

def create_evaluation_summary(all_results):
    """Create a summary comparison of all strategies"""
    df = pd.DataFrame(all_results)
    
    # Define metrics and their interpretation (higher_is_better)
    metrics_info = {
        'silhouette_score': True,
        'calinski_harabasz_score': True, 
        'davies_bouldin_score': False,
        'knn_accuracy_mean': True,
        'linear_separability_mean': True,
        'within_group_cosine_mean': True,
        'between_group_cosine_mean': False,
        'within_between_ratio': True,
        'nearest_neighbor_purity': True,
        'avg_centroid_distance': True
    }
    
    # Create ranking for each metric
    rankings = {}
    for metric, higher_is_better in metrics_info.items():
        if metric in df.columns:
            rankings[f'{metric}_rank'] = df[metric].rank(ascending=not higher_is_better)
    
    # Add rankings to dataframe
    for rank_col, ranks in rankings.items():
        df[rank_col] = ranks
    
    # Calculate overall score (average rank)
    rank_cols = [col for col in df.columns if col.endswith('_rank')]
    df['overall_rank'] = df[rank_cols].mean(axis=1)
    df['overall_score'] = len(df) + 1 - df['overall_rank']  # Convert to score
    
    return df

def create_single_strategy_plot(results, strategy_name, output_dir):
    """Create visualization for a single embedding strategy"""
    
    # Key metrics to plot
    metrics_to_plot = {
        'Silhouette Score': 'silhouette_score',
        'k-NN Accuracy': 'knn_accuracy_mean', 
        'Linear Separability': 'linear_separability_mean',
        'Within/Between Ratio': 'within_between_ratio',
        'Neighbor Purity': 'nearest_neighbor_purity',
        'Avg Centroid Distance': 'avg_centroid_distance'
    }
    
    # Filter to available metrics
    available_metrics = {k: v for k, v in metrics_to_plot.items() if v in results}
    
    if not available_metrics:
        print("No metrics available for plotting")
        return
    
    # Create subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metric_names = list(available_metrics.keys())
    metric_values = [results[available_metrics[name]] for name in metric_names]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(metric_names)), metric_values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Score')
    ax.set_title(f'Embedding Evaluation Metrics: {strategy_name}')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value + max(metric_values) * 0.01, i, f'{value:.3f}', 
                va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_evaluation_results(results_df, output_dir):
    """Create visualizations of evaluation results for multiple strategies"""
    
    # Key metrics to plot
    key_metrics = [
        'silhouette_score', 'knn_accuracy_mean', 'linear_separability_mean',
        'within_between_ratio', 'nearest_neighbor_purity', 'overall_score'
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        if metric in results_df.columns:
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(range(len(results_df)), results_df[metric])
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df['strategy'], rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            
            # Color bars by performance (green = better)
            max_val = results_df[metric].max()
            min_val = results_df[metric].min()
            for j, bar in enumerate(bars):
                normalized_val = (results_df[metric].iloc[j] - min_val) / (max_val - min_val)
                bar.set_color(plt.cm.RdYlGn(normalized_val))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create ranking heatmap
    plt.figure(figsize=(12, 8))
    
    # Get ranking columns
    rank_cols = [col for col in results_df.columns if col.endswith('_rank')]
    rank_data = results_df[['strategy'] + rank_cols].set_index('strategy')
    
    # Clean column names
    rank_data.columns = [col.replace('_rank', '').replace('_', ' ').title() for col in rank_data.columns]
    
    # Create heatmap (lower rank = better, so use reverse colormap)
    sns.heatmap(rank_data.T, annot=True, cmap='RdYlGn_r', center=len(results_df)/2, 
                cbar_kws={'label': 'Rank (1=Best)'})
    plt.title('Embedding Strategy Rankings Across Metrics')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate sentence embeddings from an npy file')
    parser.add_argument('npy_file', help='Path to the numpy embeddings file (.npy)')
    parser.add_argument('--output-dir', help='Directory to save evaluation results (defaults to same directory as npy file)')
    
    args = parser.parse_args()
    
    try:
        npy_path = Path(args.npy_file)
        if not npy_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {npy_path}")
        
        # Set output directory
        output_dir = Path(args.output_dir) if args.output_dir else npy_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using embeddings from: {npy_path}")
        print(f"Output directory: {output_dir}")
        
        # Find and load metadata
        metadata_file = find_metadata_file(npy_path)
        metadata_df = pd.read_json(metadata_file)
        metadata_df['title_id'] = metadata_df['xpath'].apply(extract_title_id)
        labels = metadata_df['title_id'].values
        
        print(f"Loaded {len(metadata_df)} samples with {len(np.unique(labels))} unique titles")
        
        # Load embeddings
        embeddings = np.load(npy_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Evaluate the embeddings
        strategy_name = npy_path.stem.replace('_embeddings', '').replace('embeddings_', '')
        results = evaluate_embedding_strategy(embeddings, labels, strategy_name)
        
        # Create a single-row dataframe for consistency with multi-strategy evaluation
        results_df = pd.DataFrame([results])
        
        # Save results
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("EMBEDDING EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nStrategy: {strategy_name}")
        print(f"Embedding dimensions: {embeddings.shape[1]}")
        print(f"Number of samples: {embeddings.shape[0]}")
        print(f"Number of unique titles: {len(np.unique(labels))}")
        
        print(f"\nEvaluation Metrics:")
        metrics_to_show = {
            'Silhouette Score': 'silhouette_score',
            'Calinski-Harabasz Score': 'calinski_harabasz_score', 
            'Davies-Bouldin Score': 'davies_bouldin_score',
            'k-NN Accuracy (mean)': 'knn_accuracy_mean',
            'Linear Separability (mean)': 'linear_separability_mean',
            'Within-Group Cosine Similarity': 'within_group_cosine_mean',
            'Between-Group Cosine Similarity': 'between_group_cosine_mean',
            'Within/Between Ratio': 'within_between_ratio',
            'Nearest Neighbor Purity': 'nearest_neighbor_purity',
            'Average Centroid Distance': 'avg_centroid_distance'
        }
        
        for display_name, metric_key in metrics_to_show.items():
            if metric_key in results:
                value = results[metric_key]
                print(f"  {display_name:<30}: {value:.4f}")
        
        # Create simple visualization for single strategy
        create_single_strategy_plot(results, strategy_name, output_dir)
        
        print(f"\nDetailed results saved to: {output_dir}/evaluation_results.csv")
        print(f"Visualization saved to: {output_dir}/evaluation_metrics.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()