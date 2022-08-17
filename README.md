# MASC

Code for Multiple Aspect  Multiple Aspect Category Sentiment Polarity Pairs Classification

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

## Running

Place all files in the same directory and run MASC_main.

## Related work

This work uses and extends the code and ideas from the following article:

Kumar, A., Gupta, P., Balan, R., Neti, L. B. M., & Malapati, A. (2021). BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment Classification. Neural Processing Letters, 53(6), 4207-4224.
