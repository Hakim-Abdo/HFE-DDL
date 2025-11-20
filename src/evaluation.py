import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self):
        pass
    
    def perform_statistical_tests(self, results_df):
        """Perform paired t-tests between models"""
        print("Performing statistical tests...")
        
        # Group by run and fold to get paired results
        paired_results = results_df.pivot_table(
            index=['run', 'fold'],
            columns='model',
            values='f1_score'
        ).reset_index()
        
        print("\nPAIRED T-TESTS (HFE-DDL vs Baselines):")
        print("=" * 50)
        
        for baseline in ['code2vec', 'word2vec', 'codebert']:
            if baseline in paired_results.columns and 'hfe_ddl' in paired_results.columns:
                hfe_scores = paired_results['hfe_ddl'].dropna()
                baseline_scores = paired_results[baseline].dropna()
                
                # Ensure same length
                min_len = min(len(hfe_scores), len(baseline_scores))
                hfe_scores = hfe_scores[:min_len]
                baseline_scores = baseline_scores[:min_len]
                
                t_stat, p_value = stats.ttest_rel(hfe_scores, baseline_scores)
                
                print(f"\nHFE-DDL vs {baseline.upper()}:")
                print(f"  T-statistic: {t_stat:.4f}")
                print(f"  P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    mean_diff = np.mean(hfe_scores) - np.mean(baseline_scores)
                    if mean_diff > 0:
                        print(f"  → HFE-DDL significantly outperforms {baseline} (p < 0.05)")
                    else:
                        print(f"  → {baseline} significantly outperforms HFE-DDL (p < 0.05)")
                else:
                    print(f"  → No significant difference")
    
    def load_and_combine_results(self):
        """Load all results files"""
        try:
            code2vec = pd.read_csv('../outputs/code2vec_results.csv')
            word2vec = pd.read_csv('../outputs/word2vec_results.csv')
            codebert = pd.read_csv('../outputs/codebert_results.csv')
            hfe_ddl = pd.read_csv('../outputs/hfe_ddl_results.csv')
            
            all_results = pd.concat([code2vec, word2vec, codebert, hfe_ddl])
            return all_results
        except FileNotFoundError as e:
            print(f"Results files not found: {e}")
            return None
    
    def plot_results_comparison(self, results_df):
        """Create comparison plots"""
        # Average results by model
        avg_results = results_df.groupby('model').agg({
            'f1_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        
        print("\nAverage Results by Model:")
        print(avg_results)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # F1 Score comparison
        plt.subplot(1, 2, 1)
        models = results_df['model'].unique()
        f1_means = [results_df[results_df['model'] == model]['f1_score'].mean() for model in models]
        f1_stds = [results_df[results_df['model'] == model]['f1_score'].std() for model in models]
        
        plt.bar(models, f1_means, yerr=f1_stds, capsize=5, alpha=0.7)
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('F1 Score')
        
        # Accuracy comparison
        plt.subplot(1, 2, 2)
        acc_means = [results_df[results_df['model'] == model]['accuracy'].mean() for model in models]
        acc_stds = [results_df[results_df['model'] == model]['accuracy'].std() for model in models]
        
        plt.bar(models, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
        plt.title('Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()