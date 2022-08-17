# MASC

Code for Multiple Aspect  Multiple Aspect Category Sentiment Polarity Pairs Classification

Implementation in Python using the PyTorch framework

## Installation

Download all files and install the packages from requirements.txt

## Files 

The following two files can be altered and run:

1. MASC_main.py       - main execution file
2. config.py          - file with configurations for the methods

The following files are called by MASC_main and do not need to be altered:

3. vocab_generator.py - category lexicon contruction
4. extractor.py       - opinion and aspect term extraction
5. score_computer.py  - category score computer
6. labeler.py         - labelling of unstructured data
7. trainer.py         - file with the training and evaluation procedure for the neural classifier
8. model.py           - classifier architecture 
9. filter_words.py    - filter words used in the method

## Datasets

Prepared datasets are available under the datasets/ directory, where you'll find two domains (laptop & restaurant) with two aspect categories configurations each. The desired dataset can be selected by editing the domain attribute in the config.py file.

## Running

Place all files in the same directory and open masc_main.py and config.py files. Configure the parameters in the latter as for e.g. domain and hardware (GPU acceleration with "CUDA" is recommended). Then you can run the masc_main.py file: first run the import lines and then the desired code fragment, e.g. run the first block in the main section for a single run using the hyperparameter setting in config.py, alternatively, run the LDC or classifier hyperopt block to perform hyperparamater optimisation using the bayesian parameter space in config.py.

## Related work

This work uses and extends the code and ideas from the following article:

Kumar, A., Gupta, P., Balan, R., Neti, L. B. M., & Malapati, A. (2021). BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment Classification. Neural Processing Letters, 53(6), 4207-4224.
