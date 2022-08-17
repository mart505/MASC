

"""# Install & imports"""

!pip install transformers
!pip install tokenizers
!pip install numba

import os
import torch
from tqdm import tqdm, notebook
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, BertForMaskedLM
import itertools
import operator
import json
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import fmin, Trials, tpe, hp

#import other python files
from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from labeler import Labeler
from trainer import Trainer
from model import BERTLinear_MASC, BERTLinear_CASC
from config import *
from filter_words import filter_words

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

domain = config["domain"]
data_path = path_mapper[domain]
print(f"domain: {domain}")

name_attempt = "it1"
name_method = "MASC"

"""# Main Code"""

vocabGenerator = VocabGenerator()
aspect_vocabularies, sentiment_vocabularies = vocabGenerator()

extracter = Extracter()
sentences, aspects, opinions = extracter()

# if vocab already generated etc., obtain from txt file
aspect_vocabularies, sentiment_vocabularies = VocabGenerator.get_from_file()
sentences, aspects, opinions = Extracter.get_from_file()

scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
scoreComputer(sentences, aspects, opinions)

labeler = Labeler()
hyp_space = hyperpar_labeler_mapper[domain]
labeler(sentences, aspects, opinions, name_attempt, hyp_space)

# train classifier (single run)

trainer = Trainer(name_method)
trainer.load_training_data_MASC(name_attempt)
hyperparameters = hyperpar_classifier_mapper[domain]
loss = trainer.train_model_MASC(hyperparameters)
trainer.save_model(name_attempt) 
#trainer.load_model(name_attempt)
F1 = trainer.evaluate_MASC()

# hyperopt Classifier

trainer = Trainer(name_method)
trainer.load_training_data_MASC(name_attempt)
best_params, loss = dict(), dict()
trials = Trials()
best_params = fmin(fn=trainer.train_model_MASC,
          space=bayes_parameters_class,
          algo=tpe.suggest,
          trials=trials,
          return_argmin=False, 
          max_evals=NUM_EVALS,
          rstate= np.random.RandomState(RANDOM_STATE))
print(best_params)

#hyperopt LDC

best_params, loss = dict(), dict()
trials = Trials()
best_params = fmin(fn=trainer.LDC_hyper,
          space=bayes_parameters_LDC,
          algo=tpe.suggest,
          trials=trials,
          return_argmin=False, 
          max_evals=NUM_EVALS,
          rstate= np.random.RandomState(RANDOM_STATE))
print(best_params)

"""Other methods"""

#CASC
name_method = "CASC"

#vocab and extraction same as MASC

scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
scoreComputer.CASC(sentences, aspects, opinions)

labeler = Labeler()
hyp_space = hyperpar_labeler_mapper[domain]
labeler.CASC(sentences, aspects, opinions, name_attempt)

trainer = Trainer(name_method)
trainer.load_training_data_CASC(name_attempt)
trainer.train_model_CASC()
trainer.save_model(name_attempt)
#trainer.load_model(name_attempt)
trainer.evaluate_CASC()

# Exp_Sim & CoSIM
name_method = "ExpSim"

trainer = Trainer(name_method)
trainer.evaluate_ExpSim(True)  # CoSIM
trainer.evaluate_ExpSim(False) # Exp_Sim