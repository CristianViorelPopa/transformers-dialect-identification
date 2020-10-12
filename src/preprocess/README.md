# Source Code - preprocess

The code here is used for preprocessing the MOROCO dataset, cleaning the documents and making it easy to fine-tune transformer models on them.

Only 2 of the scripts are supposed to be run by themselves, namely:
- `__main.py__` - real pre-processing that we used for the experiments
- `preprocess_without_labels.py` - pre-processing of unlabeled data (useful for test sets)

Both of them require only an input file (in TSV format) and where to store the processed data (in TSV format).

Aside from these, the other files offer helping functions:
- `doc_preprocessing.py` - processing of individual documents, removing artifacts and replacing symbols
- `corpus_preprocessing.py` - processing the corpus as a whole (e.g. splitting documents into sentences and using these in the final dataset)
- `constants.py` - constants defined for the preprocessing pipeline
