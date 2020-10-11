# VarDial 2020 Submission - Team Anumiți

This is the open-source code for our VarDial 2020 submission to the RDI shared task: Cristian Popa, Vlad Stefanescu. 2020. _Applying Multilingual and Monolingual Transformer-Based Models forDialect Identification_

The task was the dialect classification between Romanian and Moldavian on news extracts and tweets. Our solution focuses on comparing different transformer-based models, 3 of them multilingual and 2 of them pre-trained on the Romanian language.

The models that we experimented with are:
    - mBERT [1]
    - XLM [2]
    - XLM-R [3]
    - Cased and Uncased Romanian BERT [4]
    
The data that we used is the MOROCO corpus [5], available at https://github.com/butnaruandrei/MOROCO.

## Structure
    
We release the notebooks that we used for fine-tuning all the transformer-based models we showcase in the paper, along with the various scripts we used to preprocess data, visualize results and train ensembles.

The structure of the directories is as follows:
- `notebooks` - python notebooks for fine-tuning transformer-based models, as we used Google Colab for this endeavour
- `src` - additional scripts for everything not related to fine-tuning. Among these are: data pre-processing, analyzing and plotting results, and training of final ensemble models 
- `paper` - figures that we showcased in the paper, created using some of the scripts mentioned previously

## Citation

If this work was useful to your research, please cite it as:

Cristian Popa, Vlad Ștefănescu. 2020. _Applying Multilingual and Monolingual Transformer-Based Models for Dialect Identification_

At the moment the paper has not yet been published, so there is no link that we can provide.

## References

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. _BERT: Pre-training of deep bidirectional transformers for language understanding_. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June. Association for Computational Linguistics.

[2] Guillaume Lample and Alexis Conneau. 2019. _Cross-lingual Language Model Pretraining_. Advances in Neural Information Processing Systems (NeurIPS).

[3] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm ́an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. _Unsupervised Cross-lingual Representation Learning at Scale_. arXiv preprint arXiv:1911.02116.

[4] Stefan Daniel Dumitrescu, Andrei-Marius Avram, and Sampo Pyysalo. 2020. _The birth of Romanian BERT_.

[5] Andrei Butnaru and Radu Tudor Ionescu. 2019. _MOROCO: The Moldavian and Romanian dialectal corpus_. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 688–698, Florence, Italy, July. Association for Computational Linguistics.
