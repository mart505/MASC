from transformers import AutoTokenizer, BertForMaskedLM
from config import *
from tqdm import tqdm
import torch
from filter_words import filter_words
import numpy as np
import json

class ScoreComputer:
    '''
    Computes normalised overlap scores for each aspect category and sentiment polarity on sentence and token level and saves in "scores.txt" file
    '''
    def __init__(self, aspect_vocabularies, sentiment_vocabularies):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.mlm_model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.aspect_vocabularies = aspect_vocabularies
        self.sentiment_vocabularies = sentiment_vocabularies
    
    def __call__(self, sentences, aspects, opinions):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]
        K = K_2

        aspect_sets = self.load_vocabulary(self.aspect_vocabularies, M[self.domain])
        polarity_sets = self.load_vocabulary(self.sentiment_vocabularies, M[self.domain])
        scores_list = [ "" for i in range(len(sentences))]
        iters = 0

        for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
            aspect_words = set()
            opinion_words = set()
            if aspect != '##':
                aspect_words = set(aspect.split())
            if opinion != '##':
                opinion_words = set(opinion.split())
            ids = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
            word_predictions = self.mlm_model(ids.to(self.device))[0]
            word_scores, word_ids = torch.topk(word_predictions, K, -1)
            word_ids = word_ids.squeeze(0)
            
            token_scores = [dict() for i in range(len(tokens))]

            cntAspects = 0
            cntOpinions = 0

            for idx, token in enumerate(tokens):
                if token in aspect_words:
                    cntAspects += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for cat in categories:
                            if repl in aspect_sets[cat]:
                                token_scores[idx][cat] = token_scores[idx].get(cat, 0) + 1
                                break
                if token in opinion_words:
                    cntOpinions += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for pol in polarities:
                            if repl in polarity_sets[pol]:
                                token_scores[idx][pol] = token_scores[idx].get(pol, 0) + 1
                                break

            # create average scores
            sent_dict = dict()
            for idx, dict1 in enumerate(token_scores):
              for cat in dict1.keys():
                sent_dict[cat] = sent_dict.get(cat,0) + dict1[cat]
    
            #normalise scores for #target words
            for cat in categories:
              sent_dict[cat] = sent_dict.get(cat, 0) / max(cntAspects, 1)
            for pol in polarities:
              sent_dict[pol] = sent_dict.get(pol, 0) / max(cntOpinions, 1)
            
            # save token and sentence level scores in dict
            scores_dict = {"sentence": sent_dict, "token": token_scores}
            scores_list[iters] = scores_dict
            iters += 1
      
        sen_mean_mapper = {}
        sen_std_mapper = {}
        token_mean_mapper = {}
        token_std_mapper = {}

        #compute mean and std for sen and token level
        for cat in categories+polarities:
            category_score_list = np.array([ scores_list[i]["sentence"][cat] for i in range(len(scores_list))])
            sen_mean_mapper[cat] = np.mean(category_score_list)
            sen_std_mapper[cat] = np.std(category_score_list)

            category_score_list = np.array([ scores_list[idx1]["token"][idx2][cat] for idx1 in range(len(scores_list)) for idx2 in range(len(scores_list[idx1]["token"])) if cat in scores_list[idx1]["token"][idx2].keys()])
            token_mean_mapper[cat] = np.mean(category_score_list)
            token_std_mapper[cat] = np.std(category_score_list)

        final_dict = dict()
        for idx1, sentence in enumerate(sentences):

            #token level normalisation
            token_scores_list = []
            for _, token_scores in enumerate(scores_list[idx1]["token"]):
              for cat in token_scores.keys():
                  mean = token_mean_mapper[cat]
                  std = token_std_mapper[cat]
                  val = (token_scores.get(cat, 0) - mean) / std
                  token_scores[cat] = val 
              token_scores_list.append(token_scores)

            #sen level normalisation
            sen_scores_dict = dict()
            cat_scores = scores_list[idx1]["sentence"]
            
            for cat in categories+polarities:
                mean = sen_mean_mapper[cat]
                std = sen_std_mapper[cat]
                val = (cat_scores.get(cat, 0) - mean) / std
                sen_scores_dict[cat] = val 

            final_dict[sentence] = {"sentence": sen_scores_dict, "token": token_scores_list}

        f = open(f'{self.root_path}/scores.json', 'w')
        json.dump(final_dict, f)
        f.close()
        return

    def load_vocabulary(self, source, limit):
        target = {}
        for key in source:
            words = []
            for freq, word in source[key][:limit]:
                words.append(word)
            target[key] = set(words)
        return target

    def CASC(self, sentences, aspects, opinions):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]
        K = K_2

        aspect_sets = self.load_vocabulary(self.aspect_vocabularies, M[self.domain])
        polarity_sets = self.load_vocabulary(self.sentiment_vocabularies, M[self.domain])

        f = open(f'{self.root_path}/scores_CASC.txt', 'w')
        
        for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
            aspect_words = set()
            opinion_words = set()
            if aspect != '##':
                aspect_words = set(aspect.split())
            if opinion != '##':
                opinion_words = set(opinion.split())
            ids = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
            word_predictions = self.mlm_model(ids.to(self.device))[0]
            word_scores, word_ids = torch.topk(word_predictions, K, -1)
            word_ids = word_ids.squeeze(0)
            
            cat_scores = {}
            pol_scores = {}

            cntAspects = 0
            cntOpinions = 0

            for idx, token in enumerate(tokens):
                if token in aspect_words:
                    cntAspects += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for cat in categories:
                            if repl in aspect_sets[cat]:
                                cat_scores[cat] = cat_scores.get(cat, 0) + 1
                                break
                if token in opinion_words:
                    cntOpinions += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for pol in polarities:
                            if repl in polarity_sets[pol]:
                                pol_scores[pol] = pol_scores.get(pol, 0) + 1
                                break
            summary = f'{sentence}\n'
            for cat in categories:
                val = cat_scores.get(cat, 0) / max(cntAspects, 1)
                summary = summary + f' {cat}: {val}'
            
            for pol in polarities:
                val = pol_scores.get(pol, 0) / max(cntOpinions, 1)
                summary = summary + f' {pol}: {val}'

            f.write(summary)
            f.write('\n')
        f.close()
