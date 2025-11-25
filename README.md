# HFE-DDL: Hybrid Feature Engineering and Deep Learning for Vulnerability Detection

##  Overview

HFE-DDL (Hybrid Feature Engineering and Deep Learning) is a novel approach for Source Code vulnerability detection that combines traditional feature engineering with deep learning techniques. This repository contains the complete implementation, including data preprocessing, model training, baseline comparisons, and comprehensive evaluation.

###  Key Features

- **Hybrid Architecture**: Combines TF-IDF features with sequence embeddings using LSTM
- **Imbalanced Data Handling**: Automatic downsampling for balanced training
- **Multiple Baselines**: Comparison with Word2Vec, Code2Vec, and CodeBERT
- **Statistical Validation**: Paired t-tests and effect size analysis
- **Reproducible Experiments**: Fixed random seeds and cross-validation
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

##  Repository Structure

```
HFE-DDL-Vulnerability-Detection/
│
├── data/                          # Data directory
│   we have used LVDAndro APKs Dataset for download (https://github.com/softwaresec-labs/LVDAndro)
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── hfe_ddl_model.py          # HFE-DDL model implementation
│   ├── baselines.py              # Baseline models (Word2Vec, Code2Vec)
│   ├── evaluation.py             # Statistical testing and evaluation
│   └── utils.py                  # Utility functions
│
├── notebooks/                     # Jupyter notebooks for step-by-step execution
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── outputs/                       # Generated results and visualizations
│   ├── hfe_ddl_results.csv
│   ├── word2vec_results.csv
│   ├── code2vec_results.csv
│   ├── codebert_results.csv
│   ├── detailed_performance_metrics.csv
│   ├── comprehensive_performance_analysis.png
│   └── average_confusion_matrices.png
│
├── requirements.txt              # Python dependencies
├── run_experiment.py             # Main script to run complete experiment
└── README.md                     # This file
```

##  Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (recommended)
- NVIDIA GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HFE-DDL-Vulnerability-Detection.git
   cd HFE-DDL-Vulnerability-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for tokenization)
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage Guide

### Option 1: Run Complete Experiment (Recommended)

Execute the entire pipeline with one command:
```bash
python run_experiment.py
```

This will:
- Preprocess the data and handle class imbalance
- Train HFE-DDL model with cross-validation
- Run all baseline models
- Perform statistical tests
- Generate comprehensive results and visualizations

### Option 2: Step-by-Step Execution

For more control, run the Jupyter notebooks in order:

1. **Data Preprocessing**
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```
   - Loads and explores original data
   - Handles class imbalance through downsampling
   - Extracts TF-IDF features and prepares sequences
   - Saves preprocessed data

2. **Model Training**
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```
   - Trains HFE-DDL model with 10-fold cross-validation
   - Runs Word2Vec and Code2Vec baselines
   - Saves model results

3. **Results Analysis**
   ```bash
   jupyter notebook notebooks/03_results_analysis.ipynb
   ```
   - Performs statistical significance testing
   - Generates comprehensive visualizations
   - Provides detailed performance metrics

### Option 3: Modular Usage

Use individual components in your own code:

```python
import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from hfe_ddl_model import HFE_DDL_Model
from baselines import BaselineModels
from evaluation import Evaluator

# Preprocess data
preprocessor = DataPreprocessor()
balanced_df = preprocessor.handle_imbalanced_data("data/your_dataset.csv")

# Extract features
X_tfidf = preprocessor.extract_tfidf_features(balanced_df['processed_code'])
X_sequences = preprocessor.prepare_sequences(balanced_df['processed_code'])

# Train HFE-DDL model
hfe_ddl = HFE_DDL_Model()
model = hfe_ddl.build_model(X_tfidf.shape[1], X_sequences.shape[1])
history = hfe_ddl.train(X_tfidf, X_sequences, balanced_df['Vulnerability_status'].values)

# Run baselines
baselines = BaselineModels()
word2vec_results = baselines.run_word2vec_baseline(balanced_df)

# Evaluate results
evaluator = Evaluator()
evaluator.perform_statistical_tests(all_results)
```

##  Configuration

### Model Hyperparameters

**HFE-DDL Model:**
- TF-IDF features: 1000 dimensions
- Embedding size: 100
- LSTM units: 64
- Dense layers: 128 → 64 → 2
- Dropout: 0.3-0.4
- Regularization: L1=1e-5, L2=1e-4
- Batch size: 32
- Epochs: 15 (with early stopping)

**Baseline Models:**
- Word2Vec: 100 dimensions, window=5, skip-gram
- Code2Vec: TF-IDF with code-specific token patterns
- Random Forest: 100 estimators

### Experimental Setup
- **Cross-validation**: 10-fold stratified
- **Random seeds**: [42, 123, 456, 789, 999]
- **Evaluation metrics**: F1-score, Accuracy, Precision, Recall, Specificity

##  Results Interpretation

### Key Output Files

1. **Performance Results** (`outputs/*_results.csv`):
   - Cross-validation results for each model
   - Includes all evaluation metrics per fold

2. **Statistical Analysis**:
   - Paired t-tests between HFE-DDL and baselines
   - Effect sizes (Cohen's d)
   - Confidence intervals

3. **Visualizations**:
   - Performance comparison plots
   - Confusion matrices
   - Training curves

### Interpreting Statistical Significance

- **p-value < 0.05**: Statistically significant difference
- **Cohen's d**: Effect size (small: <0.5, medium: <0.8, large: ≥0.8)
- **Confidence intervals**: 95% confidence ranges for metrics

##  Model Architecture

### HFE-DDL Components

1. **TF-IDF Feature Branch**:
   - Input: 1000-dimensional TF-IDF vectors
   - Dense layer (128 units) with ReLU activation
   - Dropout (0.4) and regularization

2. **Sequence Branch**:
   - Input: Padded code sequences (length=15)
   - Embedding layer (vocab_size=10,000, dim=100)
   - LSTM layer (64 units) with recurrent dropout
   - Flatten and dropout layers

3. **Fusion Layer**:
   - Concatenation of both branches
   - Dense layer (64 units) with ReLU
   - Output layer (2 units) with softmax activation

##  Performance Highlights

Based on experimental results, HFE-DDL demonstrates:

- **Significant improvement** over traditional baselines (p < 0.05)
- **Robust performance** across multiple cross-validation folds
- **Balanced precision and recall** for vulnerability detection
- **Effective feature fusion** combining statistical and sequential patterns

##  Advanced Usage

### Custom Dataset

To use with your own dataset:

1. Format your data as CSV with columns:
   - `processed_code`: Preprocessed code snippets
   - `Vulnerability_status`: Binary labels (0/1)

2. Update file paths in configuration

3. Adjust preprocessing parameters if needed

### Hyperparameter Tuning

Modify hyperparameters in the respective classes:

```python
# In hfe_ddl_model.py
model = hfe_ddl.build_model(
    tfidf_shape=1000, 
    maxlen=15,
    vocab_size=10000,
    embedding_dim=100,
    lstm_units=64
)

# In baselines.py
word2vec_features = baselines.extract_word2vec_features(
    code_strings,
    vector_size=100,
    window=5,
    min_count=1
)
```

##  Troubleshooting

### Common Issues

1. **Memory Error**:
   - Reduce batch size
   - Use fewer TF-IDF features
   - Process data in chunks

2. **Missing Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

4. **CodeBERT Not Available**:
   - Install transformers: `pip install transformers torch`
   - Or skip CodeBERT baseline

### Performance Tips

- Use GPU for faster training
- Adjust sequence length based on your data
- Monitor training with TensorBoard
- Use early stopping to prevent overfitting

##  Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
gensim>=4.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

##  Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request


```



##  Acknowledgments

- Dataset: LVDAndro APKs Dataset (https://github.com/softwaresec-labs/LVDAndro)
- Libraries: TensorFlow, scikit-learn, Gensim, NLTK
- Inspiration: Code2Vec, Word2Vec, and transformer-based approaches

---
