MAX_TOKS = 512
MAX_SEQ_LEN = 512
TRAIN_SIZE = 2000
TEST_SIZE = 500

import sys
import numpy as np
import random as rn
import pandas as pd
import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

#Initialize.
rn.seed(999)
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)

#Create class that will contain test and train datasets.
class dataset:
    def __init__(self, rawData, tokenizer):
        #Turn dataframe into dict of structure [{column -> value}, … , {column -> value}].
        data = rawData.to_dict(orient = 'records')
        
        #Parse into dep/indep vars.
        self.indep = [review['text'] for review in data]
        self.dep = [review['sentiment'] for review in data]

        #Since max len is 512 and we need 2 chars for 'SEP' and 'CLS',
        #we subtract 2.
        tokenCutoff = (MAX_SEQ_LEN - 2)
        
        #Extract tokens to BERT syntax, where each token list begins
        #with '[CLS]' and ends with '[SEP]'. 
        tokensRaw = [ tokenizer.tokenize(text)[: tokenCutoff] 
            for text in self.indep ]
        
        self.tokens = [ (['[CLS]'] + tokenList + ['[SEP]']) 
            for tokenList in tokensRaw ]
        
        #Get token ids, make tensors.
        tokenIdsRaw = [ tokenizer.convert_tokens_to_ids(token)
            for token in self.tokens ]
        self.tokenIds = pad_sequences(tokenIdsRaw, maxlen = MAX_SEQ_LEN,
            truncating = 'post', padding = 'post', dtype = 'int')
        
        self.y = map(lambda el: el == 'pos', self.dep)
        
        self.masks = [
            [float(idObj > 0) for idObj in idSet]
            for idSet in self.tokenIds
        ]

#Load training/testing data as dictionaries.
trainingDataRaw = pd.read_csv('data/train.csv')[: TRAIN_SIZE]
testingDataRaw = pd.read_csv('data/test.csv')[: TEST_SIZE]

#Load BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
    do_lower_case = 1)

#Make train/test datasets.
training = dataset(trainingDataRaw, tokenizer)
testing = dataset(testingDataRaw, tokenizer)

#Initialize algorithms.
cv = CountVectorizer(ngram_range=(1,3))
logReg = LogisticRegression(max_iter = 1000)

#Create model.
model = make_pipeline(cv, logReg).fit(training.indep, training.dep)

#Test accuracy.
predicted = model.predict(testing.indep)
print(classification_report(testing.dep, predicted))