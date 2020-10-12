# Notebooks

All the notebooks were used to train the different models studied in the paper. You should be able to run them cell by cell (with some obvious modifications, such as the input files) to train the models as we did. If something breaks, note that we used the [huggingface transformers](https://github.com/huggingface/transformers) library on version 2.11.0, so we cannot guarantee for future version. They have the same exact structure, differing only in the type of pretrained model.

The notebooks are modified versions of the one developed by Chris McCormick [1].

Models used for each notebook:
- `Multilingual BERT Fine-Tuning.ipynb` - mBERT
- `XLM Fine-Tuning.ipynb` - XLM
- `XLM-RoBERTa Fine-Tuning.ipynb` - XLM-R
- `Romanian Transformer Cased Fine-Tuning.ipynb` - Cased Romanian BERT
- `Romanian Transformer Uncased Fine-Tuning.ipynb` - Uncased Romanian BERT


## References

[1] Chris McCormick and Nick Ryan. (2019, July 22). _BERT Fine-Tuning Tutorial with PyTorch_. Retrieved from http://www.mccormickml.com