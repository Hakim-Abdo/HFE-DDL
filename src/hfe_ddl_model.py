import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Flatten, Concatenate, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import random

class HFE_DDL_Model:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def build_model(self, tfidf_shape, maxlen, vocab_size=10000):
        """Build the HFE-DDL model architecture"""
        
        # TF-IDF feature branch
        input_features = Input(shape=(tfidf_shape,))
        x_features = Dense(128, activation='relu', 
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_features)
        x_features = Dropout(0.4)(x_features)
        
        # Sequence branch
        input_sequence = Input(shape=(maxlen,))
        x_sequence = Embedding(input_dim=vocab_size, output_dim=100, 
                              input_length=maxlen)(input_sequence)
        x_sequence = Dropout(0.3)(x_sequence)
        x_sequence = LSTM(64, return_sequences=True,
                         recurrent_dropout=0.2,
                         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x_sequence)
        x_sequence = Flatten()(x_sequence)
        x_sequence = Dropout(0.4)(x_sequence)
        
        # Combined branches
        concatenated = Concatenate(axis=-1)([x_features, x_sequence])
        x = Dense(64, activation='relu', 
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(concatenated)
        x = Dropout(0.4)(x)
        output_layer = Dense(2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[input_features, input_sequence], outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_tfidf, X_sequences, y, validation_data=None, epochs=15, batch_size=32):
        """Train the HFE-DDL model"""
        # Convert labels to categorical
        y_categorical = to_categorical(y, 2)
        
        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        callbacks = [early_stop]
        
        # Prepare validation data
        if validation_data:
            X_val_tfidf, X_val_seq, y_val = validation_data
            y_val_categorical = to_categorical(y_val, 2)
            val_data = ([X_val_tfidf, X_val_seq], y_val_categorical)
        else:
            val_data = None
        
        # Train model
        history = self.model.fit(
            [X_tfidf, X_sequences], y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X_tfidf, X_sequences):
        """Make predictions"""
        y_pred_proba = self.model.predict([X_tfidf, X_sequences], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred, y_pred_proba