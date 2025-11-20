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