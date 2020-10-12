# Source Code - train

The scripts here have the purpose of tuning ensemble models based on previously-tuned transformer ones. This is done by extracting the floating-point predictions of the constituents and using them to infer the final label.

Some of the scripts are meant for building the training and test corpora for the ensemble models:
- `build_ensemble_corpus.py` - with paths to predictions hard-coded inside, these are processed into features and split into train and test sets
- `build_ensemble_corpus_no_labels_no_split.py` - same as the previous script, but meant to be applied to unlabeled corpora, skipping the split into train and test sets

The rest are used strictly for tuning different types of ensemble models:
- `train_*.py` - each one tunes a simple model using grid search and saves it to disk
