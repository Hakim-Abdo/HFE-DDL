
```
HFE-DDL-Vulnerability-Detection/
│
├── dataset/                          # Data directory
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
