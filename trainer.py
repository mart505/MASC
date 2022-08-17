from transformers import AutoTokenizer, BertForMaskedLM
from config import *
from filter_words import filter_words
import torch
from tqdm import tqdm, trange, notebook
from torch.utils.data import DataLoader, TensorDataset
from model import BERTLinear_MASC, BERTLinear_CASC
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import json
from hyperopt import fmin, Trials, tpe, hp


class Trainer:

    def __init__(self, name):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.method_name = name
        self.model_name = None

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        self.model = None
        self.dataset = None

        class_dict     = {0: "positive", 1: "negative", 2: "n/a"}
        inv_class_dict = {"positive": 0, "negative": 1, "n/a": 2}

        self.class_dict = class_dict
        self.inv_class_dict = inv_class_dict

        #CASC
        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data_CASC(self, data_name):
        
        #open labelled data
        with open(f'{self.root_path}/labels_CASC_{data_name}.json', 'r') as f: #labelled_data_it3
          data_labelled = json.load(f)
        f.close()

        sentences = []
        cats = []
        pols = []

        for idx, data in enumerate(data_labelled):
          sentences.append(data['text'])
          cat = data['pairs'][0][0]
          pol = data['pairs'][0][1]
          cats.append(self.inv_aspect_dict[cat])
          pols.append(self.inv_polarity_dict[pol])

        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['token_type_ids'], encoded_dict['attention_mask'])
        self.dataset = dataset

    def load_training_data_MASC(self, data_name):

        #open labelled data
        with open(f"{self.root_path}/labelled_data_{data_name}.json", 'r') as f: #labelled_data_it3
          data_labelled = json.load(f)
        f.close()

        #make DICT from pairs list
        for idx, data in enumerate(data_labelled):
          pairs_list = data["pairs"]
          pairs_dict = {pair[0]: pair[1] for _, pair in enumerate(pairs_list)}
          data_labelled[idx]["pairs"] = pairs_dict

        #initialise vars
        pol_list, input_list = [],[]
        counter_dict = {}

        #format data for tokenizer
        for idx, data in enumerate(data_labelled):
          for cat in data["pairs"].keys():
            input_list.append((cat,data['text']))
            pol = data['pairs'][cat]
            pol_list.append(self.inv_class_dict[pol])
            counter_dict[pol] = counter_dict.get(pol, 0) + 1
        labels = torch.tensor(pol_list)

        #tokenise and create dataset object
        encoded_dict = self.tokenizer(
            input_list,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        dataset = TensorDataset(
            labels, encoded_dict['input_ids'], encoded_dict['token_type_ids'], encoded_dict['attention_mask'])
        
        #print stats
        print(f"{self.domain} train data stats: {counter_dict}")

        self.dataset = dataset
  
    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def LDC_hyper(hyp_space):
        labeler = Labeler()
        lam_label = hyp_space["threshold"]
        lam_na = hyp_space["threshold_na"]
        lam_attn = hyp_space["threshold_attn"]
        name_attempt = f"label_{lam_label}_na_{lam_na}_attn{lam_attn}"
        print(name_attempt )

        labeler(sentences, aspects, opinions, name_attempt, hyp_space)
        #trainer = Trainer(name_method)
        self.load_training_data_MASC(name_attempt)
        hyperparameters = hyperpar_classifier_mapper[domain]
        loss = self.train_model_MASC(hyperparameters)
        print(f"loss: {loss}")
        self.save_model(name_attempt)
        
        F1 = self.evaluate_MASC()
        print(f"F1 pairs: {F1}")
        return -F1

    def train_model_MASC(self, space):

        #set hyperparameters
        batch_size = int(space["batch_size"])
        #drop_out_rate = space["drop_out_rate"]
        learning_rate = space['lr']
        epochs = int(space['epochs'])
        validation_data_size = int(0.2*len(self.dataset))  # 80/20 split
        print(f"\nHyperparameters: {space}")
        
        #initialise model
        self.model = BERTLinear_MASC(self.bert_type, 3).to(self.device)
        self.set_seed(0)
        
        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            self.dataset, [len(self.dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=batch_size)
        #dataloop = notebook.tqdm(dataloader, position=0, leave=True, colour='green')
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        model = self.model
        device = self.device

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epochloop = notebook.tqdm(range(epochs), position=0, leave=True, colour='blue')
        
        for epoch in epochloop:
            dataloop = notebook.tqdm(dataloader, position=0, leave=True, colour='blue')
            model.train()
            print_loss = 0
            batch_loss = 0
            cnt = 0
            for labels, input_ids, token_type_ids, attention_mask in dataloop:
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _ = model(labels.to(device), **encoded_dict)
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1
                if cnt % 50 == 0:
                    dataloop.set_postfix(loss=batch_loss/50, data="train", epochs=epoch+1, lr=learning_rate)
                    batch_loss = 0
            print_loss /= cnt
            model.eval()
            with torch.no_grad():
                val_loss = 0
                iters = 0
                for labels, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, _ = model(labels.to(
                        device), **encoded_dict)
                    val_loss += loss.item()
                    iters += 1
                val_loss /= iters
            # Display the epoch training loss and validation loss
            epochloop.set_postfix(train_loss=print_loss, val_loss = val_loss, epochs=epoch+1)
            
            print("epoch : {:4}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(
                 epoch + 1, epochs, print_loss, val_loss))

        return val_loss

    def train_model_CASC(self):
        self.set_seed(0)

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]
        self.model = BERTLinear_CASC(self.bert_type, len(categories), len(polarities)).to(self.device)
        
        dataset = self.dataset
        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)


        model = self.model
        device = self.device

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in trange(epochs):
            model.train()
            print_loss = 0
            batch_loss = 0
            cnt = 0
            for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in dataloader:
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1
                if cnt % 50 == 0:
                    print('Batch loss:', batch_loss / 50)
                    batch_loss = 0

            print_loss /= cnt
            model.eval()
            with torch.no_grad():
                val_loss = 0
                iters = 0
                for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, _, _ = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss += loss.item()
                    iters += 1
                val_loss /= iters
            # Display the epoch training loss and validation loss
            print("epoch : {:4}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(
                epoch + 1, epochs, print_loss, val_loss))

    def save_model(self, name):
        torch.save(self.model, f'{self.root_path}/{self.method_name}_{name}.pth')
        self.model_name = name

    def load_model(self, name):
        self.model = torch.load(f'{self.root_path}/{self.method_name}_{name}.pth')
        self.model_name = name

    def evaluate_MASC(self):
        test_sentences = []
        test_classes = []
        test_cats = []
        grouped_sentences = []
        previous_sen = ""

        with open(f'{self.root_path}/test_MASC.txt', 'r') as f:
            for line in f:
                _, class_, cat, sentence = line.strip().split('\t')
                class_ = int(class_)
                test_classes.append(class_)
                test_cats.append(cat)
                test_sentences.append(sentence)
                if (sentence != previous_sen):
                  grouped_sentences.append(sentence)
                previous_sen = sentence

        df = pd.DataFrame(columns=(
            ['sentence_ID', 'sentence', 'category', 'actual class', 'predicted class', 'actual pair', 'predicted pair']))# 'actual polarity', 'predicted polarity']))

        model = self.model
        model.eval()
        device = self.device

        actual_class = []
        predicted_class = []
        actual_pair = []
        predicted_pair = []

        iters = 0
        sentence_ID = 0
        prev_sentence = test_sentences[0]
        test_loop = notebook.tqdm(zip(test_sentences, test_cats, test_classes), position=0, leave=False, colour='orange')
        with torch.no_grad():
            for sentence, cat, class_ in test_loop:
                if prev_sentence != sentence:
                  sentence_ID += 1
                prev_sentence = sentence
                
                input_ = (cat, sentence)
                encoded_dict = self.tokenizer([input_],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_class = model(torch.tensor([class_]).to(
                    device), **encoded_dict)

                actual_class.append(self.class_dict[class_])
                actual_pair.append([cat+"-"+actual_class[-1]])
                
                predicted_class.append(
                    self.class_dict[torch.argmax(logits_class).item()])
                predicted_pair.append([cat+"-"+predicted_class[-1]])
      
                df.loc[iters] = [sentence_ID, sentence, cat, actual_class[-1], predicted_class[-1], actual_pair[-1], predicted_pair[-1]]
                
                iters += 1
        

        df.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}.csv')

        # label-based measures
        predicted = np.array(predicted_class)
        actual = np.array(actual_class)

        print("\n")
        print(f"Results for {self.domain}\n")
        print("Label-based measures:")
        print("-------------------------")
        print(classification_report(actual, predicted, digits=4))
        print("-------------------------")
        print()

        # pair-based measures
        df_grouped = df.iloc[:, [0,5,6]].groupby(by="sentence_ID").sum()
        
        def remove_na(pairs_in):
          pairs_in = [pair if pair.split("-")[1] != "keyboard" else pair.split("-")[0] + pair.split("-")[1] + "-" + pair.split("-")[2] for idx,pair in enumerate(pairs_in)]
          pairs_list = [pair for idx, pair in enumerate(pairs_in) if pair.split("-")[1] != "n/a"]
          pairs_out = " ".join(pairs_list)
          return pairs_out

        df_grouped["actual pair"] = df_grouped["actual pair"].apply(remove_na)
        df_grouped["predicted pair"] = df_grouped["predicted pair"].apply(remove_na)
        df_grouped["sentence"] = grouped_sentences
        #df_grouped.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}_pairs.csv')

        # TP, FP, FN = 0, 0, 0
        # for idx, row in df_grouped.iterrows():
        #   actual_pairs = set(row["actual pair"].split())
        #   predicted_pairs = set(row["predicted pair"].split())
          
        #   TP += sum([int(pair in actual_pairs) for pair in predicted_pairs])
        #   FP += sum([int(pair not in actual_pairs) for pair in predicted_pairs])
        #   FN += sum([int(pair not in predicted_pairs) for pair in actual_pairs])

        # precision = TP / (TP + FP)
        # recall    = TP / (TP + FN)
        # F1        = 2*(precision*recall)/(precision+recall)

        # print("Pair-based measures:")
        # print("-------------------------")
        # print(f"TP: {TP}")
        # print(f"Precision: {100*precision:.2f}")
        # print(f"Recall:    {100*recall:.2f}")
        # print(f"F1:        {100*F1:.2f}")
        # print(f"string for table: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
        # print("-------------------------")

        # Aspect category measures
        # TP, FP, FN = 0, 0, 0
        # for idx, row in df_grouped.iterrows():
        #   actual_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["actual pair"].split())])
        #   predicted_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["predicted pair"].split())])
          
        #   TP += sum([int(pair in actual_aspcat) for pair in predicted_aspcat])
        #   FP += sum([int(pair not in actual_aspcat) for pair in predicted_aspcat])
        #   FN += sum([int(pair not in predicted_aspcat) for pair in actual_aspcat])

        # precision = TP / (TP + FP)
        # recall    = TP / (TP + FN)
        # F1        = 2*(precision*recall)/(precision+recall)

        # print("\n")
        # print("Aspect category measures:")
        # print("-------------------------")
        # print(f"Precision: {100*precision:.2f}")
        # print(f"Recall:    {100*recall:.2f}")
        # print(f"F1:        {100*F1:.2f}")
        # print(f"string for table: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
        # print("-------------------------")
        # print("\n")

        def evaluate_pairs(df1, name):
          TP, FP, FN = 0, 0, 0
          for idx, row in df1.iterrows():
            actual_pairs = set(row["actual pair"].split())
            predicted_pairs = set(row["predicted pair"].split())
            
            TP += sum([int(pair in actual_pairs) for pair in predicted_pairs])
            FP += sum([int(pair not in actual_pairs) for pair in predicted_pairs])
            FN += sum([int(pair not in predicted_pairs) for pair in actual_pairs])

          precision = TP / (TP + FP)
          recall    = TP / (TP + FN)
          if(precision + recall != 0):
            F1        = 2*(precision*recall)/(precision+recall)
          else:
            F1 = 0

          print(f"Pair-based measures {name}:")
          print("-------------------------")
          print(f"TP: {TP} vs. num sen: {len(df1)}")
          print(f"Precision: {100*precision:.2f}")
          print(f"Recall:    {100*recall:.2f}")
          print(f"F1:        {100*F1:.2f}")
          print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
          print("-------------------------")
          return F1

        def evaluate_cats(df1, name):
          TP, FP, FN = 0, 0, 0
          for idx, row in df1.iterrows():
            actual_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["actual pair"].split())])
            predicted_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["predicted pair"].split())])
          
            TP += sum([int(pair in actual_aspcat) for pair in predicted_aspcat])
            FP += sum([int(pair not in actual_aspcat) for pair in predicted_aspcat])
            FN += sum([int(pair not in predicted_aspcat) for pair in actual_aspcat])

          precision = TP / (TP + FP)
          recall    = TP / (TP + FN)
          if(precision + recall != 0):
            F1        = 2*(precision*recall)/(precision+recall)
          else:
            F1 = 0
          
          print(f"Aspect category measures {name}:")
          print("-------------------------")
          print(f"TP: {TP} vs. num sen: {len(df1)}")
          print(f"Precision: {100*precision:.2f}")
          print(f"Recall:    {100*recall:.2f}")
          print(f"F1:        {100*F1:.2f}")
          print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
          print("-------------------------")

        F1 = evaluate_pairs(df_grouped, "")
        evaluate_cats(df_grouped, "")
        print("\n")

        df_grouped["num_true"] = [len(pairs.split()) for pairs in df_grouped["actual pair"]]
        df_grouped.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}_pairs.csv')

        max_pairs = max(df_grouped["num_true"])
        for i in range(1,max_pairs+1):
          df_new = df_grouped[df_grouped["num_true"] == i]
          if (len(df_new) > 0):
            F1_a = evaluate_pairs(df_new, f"{i} pairs")
            evaluate_cats(df_new, f"{i} pairs")
        #return df_grouped
        return F1
        
    def evaluate_CASC(self):
        test_sentences = []
        test_pairs = []
        test_cats = []
        test_pols = []

        # with open(f'{self.root_path}/test.txt', 'r') as f:
        #     for line in f:
        #         _, cat, pol, sentence = line.strip().split('\t')
        #         cat = int(cat)
        #         pol = int(pol)
        #         test_cats.append(cat)
        #         test_pols.append(pol)
        #         test_sentences.append(sentence)

        df = pd.DataFrame(columns=(
            ['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity']))


        with open(f'{self.root_path}/test_CASC.txt', 'r') as f:
            for line in f:
                _, true_pairs, sentence = line.strip().split('\t')
                true_pairs = true_pairs.replace("'","").replace("[","").replace("]","").split(",")
                true_pairs = " ".join(true_pairs)
                test_pairs.append(true_pairs)
                test_sentences.append(sentence)

        df1 = pd.DataFrame(columns=(
            ['sentence', 'actual pair', 'predicted pair']))


        model = self.model
        model.eval()
        device = self.device

        actual_aspect = []
        predicted_aspect = []

        actual_polarity = []
        predicted_polarity = []

        predicted_pair = []

        iters = 0
        with torch.no_grad():
            for input in tqdm(test_sentences):
            #for input, cat, pol in tqdm(zip(test_sentences, test_cats, test_pols)):
                cat = 0
                pol = 0
                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([cat]).to(
                    device), torch.tensor([pol]).to(device), **encoded_dict)

                actual_aspect.append(self.aspect_dict[cat])
                actual_polarity.append(self.polarity_dict[pol])

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])

                predicted_pair.append(
                  str(self.aspect_dict[torch.argmax(logits_cat).item()]+"-"+
                  self.polarity_dict[torch.argmax(logits_pol).item()]))
                #df.loc[iters] = [input, actual_aspect[-1], predicted_aspect[-1],
                #                  actual_polarity[-1], predicted_polarity[-1]]
                df1.loc[iters] = [input, test_pairs[iters], predicted_pair[-1]]
                iters += 1

        #df.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}.csv')
        df1.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}_pairs.csv')

        def evaluate_pairs(df1, name):
          TP, FP, FN = 0, 0, 0
          for idx, row in df1.iterrows():
            actual_pairs = set(row["actual pair"].split())
            predicted_pairs = set(row["predicted pair"].split())
            
            TP += sum([int(pair in actual_pairs) for pair in predicted_pairs])
            FP += sum([int(pair not in actual_pairs) for pair in predicted_pairs])
            FN += sum([int(pair not in predicted_pairs) for pair in actual_pairs])

          precision = TP / (TP + FP)
          recall    = TP / (TP + FN)
          if(precision + recall != 0):
            F1        = 2*(precision*recall)/(precision+recall)
          else:
            F1 = "n/a"

          print(f"Pair-based measures {name}:")
          print("-------------------------")
          print(f"TP: {TP} vs. num sen: {len(df1)}")
          print(f"Precision: {100*precision:.2f}")
          print(f"Recall:    {100*recall:.2f}")
          print(f"F1:        {100*F1:.2f}")
          print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
          print("-------------------------")

        def evaluate_cats(df1, name):
          TP, FP, FN = 0, 0, 0
          for idx, row in df1.iterrows():
            actual_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["actual pair"].split())])
            predicted_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["predicted pair"].split())])
          
            TP += sum([int(pair in actual_aspcat) for pair in predicted_aspcat])
            FP += sum([int(pair not in actual_aspcat) for pair in predicted_aspcat])
            FN += sum([int(pair not in predicted_aspcat) for pair in actual_aspcat])

          precision = TP / (TP + FP)
          recall    = TP / (TP + FN)
          F1        = 2*(precision*recall)/(precision+recall)

          
          print(f"Aspect category measures {name}:")
          print("-------------------------")
          print(f"TP: {TP} vs. num sen: {len(df1)}")
          print(f"Precision: {100*precision:.2f}")
          print(f"Recall:    {100*recall:.2f}")
          print(f"F1:        {100*F1:.2f}")
          print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
          print("-------------------------")

        evaluate_pairs(df1, "")
        evaluate_cats(df1, "")
        print("\n")

        df1["num_true"] = [len(pairs.split()) for pairs in df1["actual pair"]]
        #df1.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}_pairs.csv')

        max_pairs = max(df1["num_true"])
        for i in range(1,max_pairs+1):
          df_new = df1[df1["num_true"] == i]
          if (len(df_new) > 0):
            evaluate_pairs(df_new, f"{i} pairs")
            evaluate_cats(df_new, f"{i} pairs")


    def evaluate_ExpSim(self, FLAG):
        #if flag == true, then coSim, otherwhise ExpSim
        if (self.method_name != "ExpSim"):
          return "error, wrong model"

        device = self.device
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]
        asp_seeds = aspect_seed_mapper[self.domain]
        pol_seeds = sentiment_seed_mapper[self.domain]
        bert_type = bert_mapper[self.domain]
        device = config['device']
        mlm_model = BertForMaskedLM.from_pretrained(bert_type, output_hidden_states=True).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        root_path = path_mapper[self.domain]
        mlm_model.eval()

        category_emb = {}
        for category in categories:
          seed_str = str(asp_seeds[category]).replace("{","").replace("}","").replace(",","").replace("'","")
          ids = tokenizer(seed_str, return_tensors='pt', truncation=True)['input_ids']
          tokens = tokenizer.convert_ids_to_tokens(ids[0])
          word_predictions = mlm_model(ids.to(device))["hidden_states"][11].squeeze()
          num_word = word_predictions.size()[0]
          category_emb[category] = torch.mean(word_predictions, dim=0)

        polarity_emb = {}
        for pol in polarities:
          seed_str = str(pol_seeds[pol]).replace("{","").replace("}","").replace(",","").replace("'","")
          ids = tokenizer(seed_str, return_tensors='pt', truncation=True)['input_ids']
          tokens = tokenizer.convert_ids_to_tokens(ids[0])
          word_predictions = mlm_model(ids.to(device))["hidden_states"][11].squeeze()
          num_word = word_predictions.size()[0]
          polarity_emb[pol] = torch.mean(word_predictions, dim=0)

        test_sentences = []
        test_pairs = []
        predicted_pairs = []
        evals = 25
        prec_pairs_all, rec_pairs_all, f1_pairs_all = [],[],[]
        prec_cats_all, rec_cats_all, f1_cats_all = [],[],[]
        scores_pairs_grouped = {}
        scores_cats_grouped = {}

        with open(f'{self.root_path}/test_CASC.txt', 'r') as f:
            for line in f:
                _, true_pairs, sentence = line.strip().split('\t')
                true_pairs = true_pairs.replace("'","").replace("[","").replace("]","").split(",")
                true_pairs = " ".join(true_pairs)
                test_pairs.append(true_pairs)
                test_sentences.append(sentence)


        for i in notebook.tqdm(range(evals),leave=False,position=0):
          df1 = pd.DataFrame(columns=(
              ['sentence', 'actual pair', 'predicted pair']))

          iters = 0
          for sentence in test_sentences:
            ids = tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
            tokens = tokenizer.convert_ids_to_tokens(ids[0])
            output_lastlayer = mlm_model(ids.to(device))["hidden_states"][11].squeeze()
            sentence_emb = torch.mean(output_lastlayer,dim=0)

            num_cat = len(categories)
            num_pairs = int(min(round(np.random.exponential(scale=1.0, size=None)),num_cat))
            if(FLAG): num_pairs = 1
            if(num_pairs>0):
              # calculate cosine similiarty category
              cosSim_scores = {}
              for cat in categories:
                cosSim_scores[cat] = torch.nn.functional.cosine_similarity(sentence_emb, category_emb[cat],dim=0)

              sorted_coSims = [[k,v] for k, v in sorted(cosSim_scores.items(), key=lambda item: item[1])]
              
              cosSim_pols = {}
              for pol in polarities:
                cosSim_pols[pol] = torch.nn.functional.cosine_similarity(sentence_emb, polarity_emb[pol],dim=0)
                
              pol = [[k,v] for k, v in sorted(cosSim_pols.items(), key=lambda item: item[1], reverse=True)][0][0]
              pairs = [sorted_coSims[i][0]+"-"+pol for i in range(num_pairs)]
              pairs = " ".join(pairs)
              predicted_pairs.append(pairs)
              df1.loc[iters] = [sentence, test_pairs[iters], predicted_pairs[-1]]
              iters += 1

          #df.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}.csv')
          # df1.to_csv(f'{self.root_path}/predictions_{self.method_name}.csv')
          
          def evaluate_pairs(df1, name):
            TP, FP, FN = 0, 0, 0
            for idx, row in df1.iterrows():
              actual_pairs = set(row["actual pair"].split())
              predicted_pairs = set(row["predicted pair"].split())
            
              TP += sum([int(pair in actual_pairs) for pair in predicted_pairs])
              FP += sum([int(pair not in actual_pairs) for pair in predicted_pairs])
              FN += sum([int(pair not in predicted_pairs) for pair in actual_pairs])

            precision = TP / (TP + FP)
            recall    = TP / (TP + FN)
            if(precision + recall != 0):
              F1        = 2*(precision*recall)/(precision+recall)
            else:
              F1 = 0

            return precision, recall, F1

            # print(f"Pair-based measures {name}:")
            # print("-------------------------")
            # print(f"TP: {TP} vs. num sen: {len(df1)}")
            # print(f"Precision: {100*precision:.2f}")
            # print(f"Recall:    {100*recall:.2f}")
            # print(f"F1:        {100*F1:.2f}")
            # print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
            # print("-------------------------")

          def evaluate_cats(df1, name):
            TP, FP, FN = 0, 0, 0
            for idx, row in df1.iterrows():
              actual_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["actual pair"].split())])
              predicted_aspcat = set([item.split("-")[0] for idx,item in enumerate(row["predicted pair"].split())])
            
              TP += sum([int(pair in actual_aspcat) for pair in predicted_aspcat])
              FP += sum([int(pair not in actual_aspcat) for pair in predicted_aspcat])
              FN += sum([int(pair not in predicted_aspcat) for pair in actual_aspcat])

            precision = TP / (TP + FP)
            recall    = TP / (TP + FN)
            if(precision + recall != 0):
              F1 = 2*(precision*recall)/(precision+recall)
            else:
              F1 = 0

            return precision, recall, F1

            # print(f"Aspect category measures {name}:")
            # print("-------------------------")
            # print(f"TP: {TP} vs. num sen: {len(df1)}")
            # print(f"Precision: {100*precision:.2f}")
            # print(f"Recall:    {100*recall:.2f}")
            # print(f"F1:        {100*F1:.2f}")
            # print(f"string: {100*precision:.2f} {100*recall:.2f} {100*F1:.2f}")
            # print("-------------------------")

          prec, rec, f1 = evaluate_pairs(df1, "")
          prec_pairs_all.append(prec), rec_pairs_all.append(rec), f1_pairs_all.append(f1)
          prec, rec, f1 = evaluate_cats(df1, "")
          prec_cats_all.append(prec), rec_cats_all.append(rec), f1_cats_all.append(f1)

          df1["num_true"] = [len(pairs.split()) for pairs in df1["actual pair"]]
          df1.to_csv(f'{self.root_path}/predictions_{self.method_name}_{self.model_name}_pairs.csv')

          max_pairs = max(df1["num_true"])
          for i in range(1,max_pairs+1):
            df_new = df1[df1["num_true"] == i]
            if (len(df_new) > 0):
              scores_pairs_grouped[i] = {"prec": [], "rec": [], "f1": []}
              prec, rec, f1 = evaluate_pairs(df_new, f"{i} pairs")
              scores_pairs_grouped[i]["prec"].append(prec)
              scores_pairs_grouped[i]["rec"].append(rec)
              scores_pairs_grouped[i]["f1"].append(f1)
              scores_cats_grouped[i] = {"prec": [], "rec": [], "f1": []}
              prec, rec, f1 = evaluate_cats(df_new, f"{i} pairs")
              scores_cats_grouped[i]["prec"].append(prec)
              scores_cats_grouped[i]["rec"].append(rec)
              scores_cats_grouped[i]["f1"].append(f1)
        
        #print
        def print_scores(precision, recall, f1, name):
            print(f"{name}:")
            print("-------------------------")
            print(f"Precision: {100*precision:.2f}")
            print(f"Recall:    {100*recall:.2f}")
            print(f"F1:        {100*f1:.2f}")
            print(f"string: {100*precision:.2f} {100*recall:.2f} {100*f1:.2f}")
            print("-------------------------")
        
        #all
        prec = np.average(prec_pairs_all)
        rec = np.average(rec_pairs_all)
        f1 = np.average(f1_pairs_all)
        print_scores(prec, rec, f1, "Pair-based measures overall")
        prec = np.average(prec_cats_all)
        rec = np.average(rec_cats_all)
        f1 = np.average(f1_cats_all)
        print_scores(prec, rec, f1, "cat-based measures overall")

        #grouped
        for i in range(1, max_pairs+1):
          if i in scores_pairs_grouped.keys():
            prec = np.average(scores_pairs_grouped[i]["prec"])
            rec = np.average(scores_pairs_grouped[i]["rec"])
            f1 = np.average(scores_pairs_grouped[i]["f1"])
            print_scores(prec, rec, f1, f"Pair-based measures {i} pairs")
            prec = np.average(scores_cats_grouped[i]["prec"])
            rec = np.average(scores_cats_grouped[i]["rec"])
            f1 = np.average(scores_cats_grouped[i]["f1"])
            print_scores(prec, rec, f1, f"cat-based measures {i} pairs")

          
