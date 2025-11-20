import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Union, Optional

class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.tokenizer = None
        
    def handle_imbalanced_data(self, file_path, target_column='Vulnerability_status'):
        """Handle imbalanced data through downsampling"""
        print("Loading and balancing dataset...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # Separate classes
        df_majority = df[df[target_column] == 0]
        df_minority = df[df[target_column] == 1]
        
        # Downsample majority class
        n_minority = len(df_minority)
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=n_minority,
            random_state=self.random_state
        )
        
        # Combine and shuffle
        balanced_df = pd.concat([df_majority_downsampled, df_minority])
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Balanced dataset shape: {balanced_df.shape}")
        print(f"Class distribution after balancing:\\n{balanced_df[target_column].value_counts()}")
        
        return balanced_df
    
    def extract_tfidf_features(self, text_data, max_features=1000):
        """Extract TF-IDF features"""
        print("Extracting TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=0.0,
            max_df=1.0
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
        feature_array = tfidf_matrix.toarray()
        
        print(f"TF-IDF features shape: {feature_array.shape}")
        return feature_array
    
    def prepare_sequences(self, text_data, num_words=10000, maxlen=15):
        """Prepare text sequences for neural network"""
        print("Preparing text sequences...")
        
        self.tokenizer = Tokenizer(
            num_words=num_words,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n',
            lower=True,
            split=' ',
            char_level=False
        )
        
        # Fit tokenizer and create sequences
        self.tokenizer.fit_on_texts(texts=text_data)
        sequences = self.tokenizer.texts_to_sequences(texts=text_data)
        
        # Analyze sequence lengths
        sequence_lengths = [len(sequence) for sequence in sequences]
        print(f"Average sequence length: {np.mean(sequence_lengths):.2f}")
        print(f"95th percentile length: {np.percentile(sequence_lengths, 95)}")
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences=sequences,
            maxlen=maxlen,
            dtype='int32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        print(f"Padded sequences shape: {padded_sequences.shape}")
        return padded_sequences