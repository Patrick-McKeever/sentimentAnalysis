MAX_TOKS = 512
MAX_SEQ_LEN = 512
TRAIN_SIZE = 2000
TEST_SIZE = 500
BATCH_SIZE = 4
EPOCHS = 5

import sys
import numpy as np
import random as rn
import pandas as pd
import torch
import pickle
from pytorch_pretrained_bert import BertModel
from torch import nn
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Used to instantiate training and testing datasets.
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
        
        tokens = [ (['[CLS]'] + tokenList + ['[SEP]']) 
            for tokenList in tokensRaw ]
        
        #Get token ids, make tensors.
        tokenIdsRaw = [ tokenizer.convert_tokens_to_ids(token)
            for token in tokens ]
        self.tokenIds = pad_sequences(tokenIdsRaw, maxlen = MAX_SEQ_LEN,
            truncating = 'post', padding = 'post', dtype = 'int')
        self.tokensTensor = torch.tensor(self.tokenIds)
        
        #Make tensors for sentiment type.
        self.y = np.array(self.dep) == 'pos'
        self.yTensor = torch.tensor(self.y.reshape(-1, 1)).float()
        
        self.x = torch.tensor(self.tokenIds[ : 3])
        
        #Make tensors for masks.
        masks = [
            [float(idObj > 0) for idObj in idSet]
            for idSet in self.tokenIds
        ]
        self.masksTensor = torch.tensor(masks)
        
#Load training/testing data as dictionaries.
trainingDataRaw = pd.read_csv('data/train.csv')[: TRAIN_SIZE]
testingDataRaw = pd.read_csv('data/test.csv')[: TEST_SIZE]

#Load BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
    do_lower_case = 1)

training = dataset(trainingDataRaw, tokenizer)
testing = dataset(testingDataRaw, tokenizer)

#Unigrams through trigrams.
cv = CountVectorizer(ngram_range = (1, 3))
logReg = LogisticRegression(max_iter = 1000)

model = make_pipeline(cv, logReg).fit(training.indep, training.dep)
pickle.dump(model, open('logReg.sav', 'wb+'))

predicted = model.predict(testing.indep)
print(classification_report(testing.dep, predicted))

#Make a neural network based on pytorch requirements.
class sentClassifier(nn.Module):
    def __init__(self):
        super(sentClassifier, self).__init__()

        self.bertModel = BertModel.from_pretrained('bert-base-uncased')
        bertSize = self.bertModel.config.hidden_size
        
        self.drop = nn.Dropout(p = 0.1)
        self.lin = nn.Linear(bertSize, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokenIds, masks):
        pooledOut = self.bertModel(tokenIds, attention_mask = masks, 
            output_all_encoded_layers = 0)[1]
        
        output = self.drop(pooledOut)
        linearOut = self.lin(output)
        return self.sigmoid(linearOut)

upgradedModel = sentClassifier()
modelOnDev = upgradedModel.cuda()

trainData = TensorDataset(training.tokensTensor,
    training.masksTensor, training.yTensor)
trainSampler = RandomSampler(trainData)
trainDataLoader = DataLoader(trainData,
    sampler = trainSampler, batch_size = BATCH_SIZE)

testData = TensorDataset(testing.tokensTensor,
    testing.masksTensor, testing.yTensor)
testSampler = SequentialSampler(testData)
testDataLoader = DataLoader(testData, 
    sampler = testSampler, batch_size = BATCH_SIZE)

#Optimize model parameters.
params = list(modelOnDev.sigmoid.named_parameters()) 
groupedParams = [{"params": [p for n, p in params]}]

optimizer = Adam(modelOnDev.parameters(), lr = 3e-6)

for epoch in range(EPOCHS):
    modelOnDev.train()
    trainLoss = 0
    
    for step, data in enumerate(trainDataLoader):
        print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')
        print(step)
        tokenIds, masks, sents = tuple(datum.cuda() for datum in data)
        logits = modelOnDev(tokenIds, masks)
        
        
        lossFunc = nn.BCELoss()

        batchLoss = lossFunc(logits, sents)
        trainLoss += batchLoss.item()
        
        modelOnDev.zero_grad()
        batchLoss.backward()
        
        clip_grad_norm_(parameters = modelOnDev.parameters(),
            max_norm = 1.0)
        optimizer.step()
        
        totalSteps = len(trainData) / BATCH_SIZE
        currentLoss = trainLoss / (step + 1)

        clear_output(wait = 1)
        print('Epoch: ', epoch + 1)
        print('\r' + '%d/%d loss: %f ' 
            % (step, totalSteps, currentLoss))

pickle.dump(modelOnDev, open('finalModel.sav', 'wb+'))

predicted = []
cumLogits = []
with torch.no_grad():
    for step, data in enumerate(testDataLoader):
        print('step %d of %d' % (step, len(testDataLoader)))
        tokenIds, masks, sents = tuple(datum.cuda() for datum in data)
        logits = modelOnDev(tokenIds, masks)
        
        lossFunc = nn.BCELoss()
        batchLoss = lossFunc(logits, sents)
        
        loss = lossFunc(logits, sents)
        numLogits = logits.cpu().detach().numpy()
        
        predicted += list(numLogits[:, 0] > 0.5)
        cumLogits += list(numLogits[:, 0])

print(classification_report(testing.y, predicted))
