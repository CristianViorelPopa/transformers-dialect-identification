# Source Code

This directory contains the Python source code we used for the work unrelated to the transformer fine-tuning.

## Structure

The source code is partitioned in the following directories:
- `preprocess` - scripts and helping modules for pre-processing the MOROCO data in TSV format
- `eval` - scripts for evaluating the performance of models and visualizing the results 
- `train` - scripts for creating the dataset from predictions of the transformer models, along with tuning various ensemble models on top of that data
- `attention-analysis` - a slightly modified version of the codebase available at https://github.com/clarkkev/attention-analysis, to fit our needs
- `paper`- scripts with a very specific purpose, used to create the figures in the paper

Each of the directories contain their own README to better explain the code and how to make use of it, except for the one pertaining to attention analysis, where we left a link to the original repo. Please cite their original work if you find it useful.
