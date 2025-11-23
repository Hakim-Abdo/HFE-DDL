# baselines.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import random
import os
import re

class BaselineModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    # Word2Vec Baseline
    def extract_word2vec_features(self, code_strings, vector_size=100, window=5, min_count=1):
        """Extract Word2Vec features"""
        print("Extracting Word2Vec features...")
        
        # Tokenize code
        tokenized_code = [word_tokenize(str(code).lower()) for code in code_strings]
        
        # Train Word2Vec model
        word2vec_model = Word2Vec(
            sentences=tokenized_code,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=10
        )
        
        # Create document embeddings
        document_embeddings = []
        for tokens in tokenized_code:
            vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
            doc_vector = np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(vector_size)
            document_embeddings.append(doc_vector)
        
        return np.array(document_embeddings)
    
    def run_word2vec_baseline(self, df, n_runs=1, n_folds=10):
        """Run Word2Vec baseline"""
        print("Running Word2Vec baseline...")
        
        # Ensure output directory exists
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/word2vec_results.csv"
        
        # Extract features
        word2vec_features = self.extract_word2vec_features(df['processed_code'])
        
        seeds = [42, 123, 456, 789, 999][:n_runs]
        file_exists = os.path.isfile(output_file)
        
        for run_idx, seed in enumerate(seeds):
            print(f"Run {run_idx+1}/{n_runs} (Seed: {seed})")
            self.set_seeds(seed)
            
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(word2vec_features, df['Vulnerability_status'])):
                X_train = word2vec_features[train_idx]
                X_test = word2vec_features[test_idx]
                y_train = df['Vulnerability_status'].values[train_idx]
                y_test = df['Vulnerability_status'].values[test_idx]
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=seed)
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                metrics.update({
                    'run': run_idx + 1,
                    'fold': fold + 1,
                    'seed': seed,
                    'model': 'word2vec'
                })
                
                # Save results incrementally
                df_result = pd.DataFrame([metrics])
                df_result.to_csv(output_file, mode='a', index=False, header=not file_exists)
                file_exists = True
                
                print(f"  Fold {fold+1}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        return pd.read_csv(output_file)
    
    # Code2Vec Baseline 
    def extract_code2vec_features(self, code_strings):
        """Extract code2vec-style features"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        token_pattern = r'''([a-zA-Z_][a-zA-Z0-9_]*|==|<=|>=|\\+|\\-|\\*|/|<|>|=|\\d+\\.\\d+|\\d+|\".*?\"|'.*?'|\\(|\\)|\\[|\\]|\\{|\\})'''
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            token_pattern=token_pattern,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        
        features = vectorizer.fit_transform(code_strings)
        return features
    
    def run_code2vec_baseline(self, df, n_runs=1, n_folds=10):
        """Run code2vec baseline"""
        print("Running code2vec baseline...")
        
        code2vec_features = self.extract_code2vec_features(df['processed_code'])
        seeds = [42, 123, 456, 789, 999][:n_runs]
        all_results = []
        
        for run_idx, seed in enumerate(seeds):
            print(f"Run {run_idx+1}/{n_runs} (Seed: {seed})")
            self.set_seeds(seed)
            
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(code2vec_features, df['Vulnerability_status'])):
                X_train = code2vec_features[train_idx]
                X_test = code2vec_features[test_idx]
                y_train = df['Vulnerability_status'].values[train_idx]
                y_test = df['Vulnerability_status'].values[test_idx]
                
                model = RandomForestClassifier(n_estimators=100, random_state=seed)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                metrics.update({
                    'run': run_idx + 1,
                    'fold': fold + 1,
                    'seed': seed,
                    'model': 'code2vec'
                })
                all_results.append(metrics)
                
                print(f"  Fold {fold+1}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('outputs/code2vec_results.csv', index=False)
        return results_df
    
    # CodeBERT Baseline
    def extract_codebert_features(self, code_strings, batch_size=8, model_name="microsoft/codebert-base"):
        """
        Extract features using CodeBERT model
        Requires: pip install transformers torch
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print(f"Loading CodeBERT model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode
            model.eval()
            
            features = []
            
            print("Extracting CodeBERT embeddings...")
            for i in range(0, len(code_strings), batch_size):
                batch_texts = code_strings[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts.tolist() if hasattr(batch_texts, 'tolist') else batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embedding as document representation
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    features.append(batch_features)
                
                if (i // batch_size) % 50 == 0:
                    print(f"  Processed {i}/{len(code_strings)} samples...")
            
            features_array = np.vstack(features)
            print(f"CodeBERT features shape: {features_array.shape}")
            return features_array

        except ImportError:
            print("Transformers or torch not available. Using TF-IDF as fallback.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=768)  # Match BERT hidden size
            features = vectorizer.fit_transform(code_strings).toarray()
            print(f"Fallback TF-IDF features shape: {features.shape}")
            return features
        
        except Exception as e:
            print(f"Error in CodeBERT feature extraction: {e}")
            print("Using TF-IDF as fallback...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=768)
            features = vectorizer.fit_transform(code_strings).toarray()
            return features
    
    def run_codebert_baseline(self, df, n_runs=1, n_folds=10, batch_size=8):
        """Run CodeBERT baseline multiple times with different seeds"""
        print("Running CodeBERT baseline...")
        
        # Ensure output directory exists
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/codebert_results.csv"
        
        print("Extracting CodeBERT features...")
        codebert_features = self.extract_codebert_features(df['processed_code'], batch_size=batch_size)
        
        seeds = [42, 123, 456, 789, 999][:n_runs]
        file_exists = os.path.isfile(output_file)
        
        for run_idx, seed in enumerate(seeds):
            print(f"Run {run_idx+1}/{n_runs} (Seed: {seed})")
            self.set_seeds(seed)
            
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(codebert_features, df['Vulnerability_status'])):
                X_train = codebert_features[train_idx]
                X_test = codebert_features[test_idx]
                y_train = df['Vulnerability_status'].values[train_idx]
                y_test = df['Vulnerability_status'].values[test_idx]
                
                # Train classifier on CodeBERT features
                # Using LogisticRegression for better performance with high-dimensional features
                model = LogisticRegression(random_state=seed, max_iter=1000, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                metrics.update({
                    'run': run_idx + 1,
                    'fold': fold + 1,
                    'seed': seed,
                    'model': 'codebert'
                })
                
                # Save results incrementally
                df_result = pd.DataFrame([metrics])
                df_result.to_csv(output_file, mode='a', index=False, header=not file_exists)
                file_exists = True
                
                print(f"  Fold {fold+1}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}, "
                      f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
        
        print(f"CodeBERT results saved to: {output_file}")
        return pd.read_csv(output_file)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        return {
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    
    def run_all_baselines(self, df, n_runs=1, n_folds=10):
        """
        Run all baseline models sequentially
        """
        print("Running all baseline models...")
        print("=" * 50)
        
        results = {}
        
        # Run Word2Vec
        print("\n1. Word2Vec Baseline")
        results['word2vec'] = self.run_word2vec_baseline(df, n_runs, n_folds)
        
        # Run Code2Vec
        print("\n2. Code2Vec Baseline")
        results['code2vec'] = self.run_code2vec_baseline(df, n_runs, n_folds)
        
        # Run CodeBERT (with error handling)
        print("\n3. CodeBERT Baseline")
        try:
            results['codebert'] = self.run_codebert_baseline(df, n_runs, n_folds)
        except Exception as e:
            print(f"CodeBERT failed: {e}")
            print("Skipping CodeBERT baseline...")
        
        print("\n" + "=" * 50)
        print("All baselines completed!")
        
        return results


# Example usage
if __name__ == "__main__":
    # Example of how to use the baselines
    baselines = BaselineModels(random_state=42)
    
    # Load your data
    # df = pd.read_csv("your_dataset.csv")
    
    # Run individual baseline
    # word2vec_results = baselines.run_word2vec_baseline(df, n_runs=1, n_folds=5)
    
    # Run all baselines
    # all_results = baselines.run_all_baselines(df, n_runs=1, n_folds=5)
    
    print("Baseline models module loaded successfully!")
    print("Available methods:")
    print("- run_word2vec_baseline()")
    print("- run_code2vec_baseline()") 
    print("- run_codebert_baseline()")
    print("- run_fasttext_baseline()")
    print("- run_all_baselines()")