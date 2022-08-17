import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from config import *

class LQLoss(nn.Module):
    
    def __init__(self, q, weight, alpha=0.0):
        super().__init__()
        self.q = q ## parameter in the paper
        self.alpha = alpha ## hyper-parameter for trade-off between weighted and unweighted GCE Loss
        self.weight = nn.Parameter(F.softmax(torch.log(1 / torch.tensor(weight)), dim=-1), requires_grad=False).to('cuda') ## per-class weights

    def forward(self, input, target, *args, **kwargs):
        bsz, _ = input.size()

        Yq = torch.gather(input, 1, target.unsqueeze(1))
        lq = (1 - torch.pow(Yq, self.q)) / self.q

        _weight = self.weight.repeat(bsz).view(bsz, -1)
        _weight = torch.gather(_weight, 1, target.unsqueeze(1))
    
        return torch.mean(self.alpha * lq + (1 - self.alpha) * lq * _weight)

class BERTLinear_CASC(nn.Module):
    def __init__(self, bert_type, num_cat, num_pol):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.ff_pol = nn.Linear(768, num_pol)
        self.aspect_weights = aspect_categories_dist[config["domain"]] # aspect category distribution
        self.sentiment_weights = sentiment_categories_dist[config["domain"]] # sentiment category distribution

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
        loss = LQLoss(0.4, self.aspect_weights)(F.softmax(logits_cat), labels_cat) + LQLoss(0.4, self.sentiment_weights)(F.softmax(logits_pol), labels_pol)
        return loss, logits_cat, logits_pol

#custom code
class BERTLinear_MASC(nn.Module):
    def __init__(self, bert_type, num_class):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_class = nn.Linear(768*2, num_class)

        self.class_weights = class_categories_dist[config["domain"]] # class category distribution (restaurant)
        #self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
    def forward(self, labels, **kwargs):
        outputs = self.bert(**kwargs)
        # last emb, only sentence
        x = outputs[2][11]
        seq_len = x.size()[1] # (bsz, seq_len, 768) bsz = batch size #last layer word embeddings
        x = x[:,3:seq_len,:]  #  remove 3 tokens for asp cat
        cat_emb = outputs[2][11][:,1,:].squeeze()
  
        mask = kwargs['attention_mask'][:,3:seq_len]  # (bsz, seq_len)
        sen_emb = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        sen_emb = sen_emb.sum(dim=1) / den  # (bsz, 768)

        if (sen_emb.size()[0] == 1): #if bsz = 1
          cat_emb = cat_emb.unsqueeze(0)
        
        final_emb = torch.cat((cat_emb,sen_emb),dim=1)
        logits_class = self.ff_class(final_emb)  # (bsz, num_cat)
  
        #loss = self.criterion(logits_class,labels)
        loss = LQLoss(0.4, self.class_weights)(F.softmax(logits_class), labels)
        return loss, logits_class #, logits_pol
