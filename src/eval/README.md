# Source Code - eval

The scripts here are used for evaluating our models in multiple ways:
- `attention_analysis.py` plots the results of our attention analysis, such as the differentiating tokens for each dialect; it is meant to use the output of the scripts in the `attention-analysis` module
- `*_predict.py` makes floating-point predictions on a dataset and can store them to disk in numpy array format (applies for both transformer-based and ensemble models)
- `*_performance.py` computes metrics, such as F1 score, and plots the ROC and confusion matrix on a dataset (applies for both transformer-based and ensemble models)
