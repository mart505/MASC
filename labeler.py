from config import *
from transformers import AutoTokenizer, BertForMaskedLM
import numpy as np
import json
import torch
from tqdm import tqdm, notebook
import operator

class Labeler:

    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.mlm_model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        
    
    def __call__(self, sentences, aspects, opinions, file_name, space):
      
      # load in the scores file
      with open(f'{self.root_path}/scores.json', 'r') as f:
          cat_dict_by_sentence = json.load(f)
      categories = aspect_category_mapper[self.domain]
      polarities = sentiment_category_mapper[self.domain]

      # hyperparameters
      threshold = space['threshold']
      threshold_na = space['threshold_na']
      threshold_attn = space['threshold_attn']
      # per sentence
      cnt = 0
      labels = []
      labels_cnt = 0
      mult_label_cnt = 0

      for sentence, aspect, opinion in notebook.tqdm(zip(sentences, aspects, opinions)):
          cnt += 1
          scores_dict = cat_dict_by_sentence[sentence]["sentence"]
          token_scores_list = cat_dict_by_sentence[sentence]["token"]
          
          # identify aspect categories with > 0.5 score
          potential_cats = [k for (k,v) in scores_dict.items() if v > threshold and k in categories]

          if aspect != '##':
            aspect_words = set(aspect.split())
            aspect_words = { asp.lower() for asp in aspect_words}
          if opinion != '##':
            opinion_words = set(opinion.split())
            opinion_words = { opin.lower() for opin in opinion_words}

          ids = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
          tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
          word_predictions = self.mlm_model(ids.to(self.device), output_attentions=True)[0]
          # get average attention over all attn layers & all heads
          word_attentions = self.mlm_model(ids.to(self.device), output_attentions=True)[1]
          word_attention_summed = sum(word_attentions)/12
          word_attention_avg = torch.sum(word_attention_summed[0], 0)/12
          
          #remove sep tokens and redistribute attentions
          word_attn = [word_attention_avg[i,1:len(tokens)-1]/ sum(word_attention_avg[i,1:len(tokens)-1]) for i in range(len(tokens))]

          token_scores = token_scores_list 
          token_idx_dict = {token: idx for idx, token in enumerate(tokens)}
        
          # assign aspect terms to potential category by highest score
          tokens_assigned = {}
          for idx, token in enumerate(tokens):
            if token in aspect_words and len(token_scores[idx].keys()) > 0:
              cat_scores = {key: value for key,value in token_scores[idx].items() if key in potential_cats}
              if (len(cat_scores.keys())> 0):
                cat_max = max(cat_scores.items(), key=operator.itemgetter(1))
                if cat_max[1] > -0.5:
                  cat = cat_max[0]
                  tokens_assigned[cat] = (tokens_assigned.get(cat,"") + " " + token).strip()
          
          # go through all sentiments and see if any connections to aspect terms
          label_list = []
          num_label_cnt = 0
          for cat in tokens_assigned.keys():
            aspect_terms = set(tokens_assigned[cat].split())
            pol_scores = dict()
            opinion_targets = ""
            for idx, token in enumerate(tokens):
              if token in opinion_words:
                #threshold_attn = 1/(len(tokens)-2)
                attn1 = sum([word_attn[idx-1][token_idx_dict[asp]-1] > threshold_attn for asp in aspect_terms])>0
                attn2 = sum([word_attn[token_idx_dict[asp]-1][idx-1] > threshold_attn for asp in aspect_terms])>0
                if attn1 and attn2:
                  opinion_targets += " " + token 
                  for pol in polarities:
                    pol_scores[pol] = pol_scores.get(pol,0) + token_scores_list[idx].get(pol,0)
            if len(pol_scores.keys())>0:
              max_pol = max(pol_scores.items(), key=operator.itemgetter(1))
              if max_pol[1] > threshold:
                label_list.append([cat, max_pol[0]])
                labels_cnt += 1
                num_label_cnt += 1
          
          if (num_label_cnt > 1):
            mult_label_cnt += 1
          
          for cat in categories:
            if scores_dict[cat] < threshold_na:
              label_list.append((cat,"n/a"))
          output_dict = {"text": sentence , "pairs": label_list}
          labels.append(output_dict)
          
      print(f"cnt labels: {labels_cnt}")
      print(f"cnt sentences w/ mult labels: {mult_label_cnt}")
      with open(f'{self.root_path}/labelled_data_{file_name}.json', 'w') as f: #led_data_it3
        json.dump(labels, f)

    def CASC(self, sentences, aspects, opinions, file_name):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        # Distributions
        dist = {}
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        with open(f'{self.root_path}/scores_CASC.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    values = line.strip().split()
                    for j, val in enumerate(values):
                        if j % 2 == 1:
                            dist[values[j-1][:-1]].append(float(val))
        
        # Compute mean and sigma for each category
        means = {}
        sigma = {}
        for key in dist:
            means[key] = np.mean(dist[key])
            sigma[key] = np.std(dist[key])
        
        nf = open(f'{self.root_path}/labels_CASC_{file_name}.json', 'w')
        cnt = {}
        labels_list = []
        with open(f'{self.root_path}/scores_CASC.txt', 'r') as f:
            sentence = None
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    labels_dict = {}
                    labels_dict["text"] = sentence.strip()
                    labels_dict["pairs"] = []
                    aspect = []
                    sentiment = []
                    key = None
                    for j, val in enumerate(line.strip().split()):
                        if j % 2 == 1:
                            # Normalise score
                            dev = (float(val) - means[key]) / sigma[key]
                            if dev >= lambda_threshold:
                                if key in categories:
                                    aspect.append(key)
                                else:
                                    sentiment.append(key)
                            # comment this out if not to add n/a
                            # if dev < -0.5:
                            #     if key in categories:
                            #         labels_dict["pairs"].append([key,"n/a"])
                            #         keyword = f'{key}-n/a'
                            #         cnt[keyword] = cnt.get(keyword, 0) + 1
                        else:
                            key = val[:-1]
                    # No conflict (avoid multi-class sentences)
                    if len(aspect) == 1 and len(sentiment) == 1:
                        labels_dict["pairs"].append([aspect[0],sentiment[0]])
                        keyword = f'{aspect[0]}-{sentiment[0]}'
                        cnt[keyword] = cnt.get(keyword, 0) + 1
                        labels_list.append(labels_dict)
                else:
                    sentence = line

        json.dump(labels_list,nf)
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)