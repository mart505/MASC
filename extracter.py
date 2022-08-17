from os import linesep
from tqdm import tqdm
from config import *
import spacy


class Extracter:
    '''
    Extract potential-aspects and potential-opinion words
    '''

    def __init__(self):
        self.smodel = spacy.load("en_core_web_sm") #better version
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]

    def __call__(self):
        # Extract potential-aspects and potential-opinions
        sentences = []
        aspects = []
        opinions = []
        cnt = 0

        with open(f'{self.root_path}/train.txt') as f:
            for line in tqdm(f):
                text = line.strip()
                sentences.append(text)
                words = self.smodel(text)
                o = []
                a = []
                for word in words:
                    if word.tag_.startswith('JJ') or word.tag_.startswith('RR'):
                        # Adjective or Adverb
                        o.append(word.text)
                    if word.tag_.startswith('NN'):
                        # Noun
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        f = open(f'{self.root_path}/sentences_extracted.txt', 'w')
        for sentence in sentences:
            f.write(f'{sentence}\n')
        f.close()
        f = open(f'{self.root_path}/aspects_extracted.txt', 'w')
        for aspect in aspects:
            f.write(f'{aspect}\n')
        f.close()
        f = open(f'{self.root_path}/opinions_extracted.txt', 'w')
        for opinion in opinions:
            f.write(f'{opinion}\n')
        f.close()

        
        return sentences, aspects, opinions

    def vocab_from_file(self):

      def load_vocab(categories):
        vocabularies = dict()
        for category in categories:
          f = open(f'{self.root_path}/dict_{category}.txt', 'r')
          words = []
          for line in f:
            word, freq = line.split()
            words.append((freq,word)) 
          vocabularies[category] = words
        return vocabularies

      aspect_categories = aspect_category_mapper[self.domain]
      sentiment_categories = sentiment_category_mapper[self.domain]

      aspect_vocabularies = load_vocab(aspect_categories)
      sentiment_vocabularies = load_vocab(sentiment_categories)

      return aspect_vocabularies, sentiment_vocabularies
    
    @staticmethod
    def get_from_file():
      
      root_path = path_mapper[config['domain']]
      sentences, aspects, opinions = [],[],[]

      f = open(f'{root_path}/aspects_extracted.txt', 'r')
      for line in f:
        aspects.append(line.strip())

      f = open(f'{root_path}/opinions_extracted.txt', 'r')
      for line in f:
        opinions.append(line.strip())

      f = open(f'{root_path}/sentences_extracted.txt', 'r')
      for line in f:
        sentences.append(line.strip())

      return sentences, aspects, opinions
