import sys
import os
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from hfe_ddl_model import HFE_DDL_Model
from baselines import BaselineModels
from evaluation import Evaluator
import pandas as pd

def main():
    print("=== HFE-DDL Vulnerability Detection Experiment ===\n")
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    preprocessor = DataPreprocessor(random_state=42)
    
    # Handle imbalanced data
    balanced_df = preprocessor.handle_imbalanced_data(
        "data/LVDAndro_APKs_Combined_Processed.csv"
    )
    
    # Save balanced data
    balanced_df.to_csv('data/LVDAndro_APKs_downsampled.csv', index=False)
    
    # Extract features
    X_tfidf = preprocessor.extract_tfidf_features(balanced_df['processed_code'])
    X_sequences = preprocessor.prepare_sequences(balanced_df['processed_code'])
    y = balanced_df['Vulnerability_status'].values
    
    print(f"Final dataset shape: {balanced_df.shape}")
    print(f"TF-IDF features shape: {X_tfidf.shape}")
    print(f"Sequences shape: {X_sequences.shape}\n")
    
    # Step 2: Run Baselines
    print("Step 2: Running Baselines")
    baselines = BaselineModels(random_state=42)
    
    # Run Word2Vec baseline
    word2vec_results = baselines.run_word2vec_baseline(balanced_df, n_runs=1, n_folds=10)
    
    # Run Code2Vec baseline
    code2vec_results = baselines.run_code2vec_baseline(balanced_df, n_runs=1, n_folds=10)
    
    # Step 3: Train HFE-DDL Model
    print("\nStep 3: Training HFE-DDL Model")
    hfe_ddl = HFE_DDL_Model(random_state=42)
    model = hfe_ddl.build_model(X_tfidf.shape[1], X_sequences.shape[1])
    
    print("Model architecture summary:")
    model.summary()
    
    # Step 4: Evaluation
    print("\nStep 4: Evaluation and Statistical Testing")
    evaluator = Evaluator()
    
    # Load all results
    all_results = evaluator.load_and_combine_results()
    if all_results is not None:
        # Perform statistical tests
        evaluator.perform_statistical_tests(all_results)
        
        # Create comparison plots
        evaluator.plot_results_comparison(all_results)
    
    print("\n=== Experiment Complete ===")

if __name__ == "__main__":
    main()