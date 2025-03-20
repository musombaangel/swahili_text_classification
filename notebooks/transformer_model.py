import torch
import torch.nn as nn
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW

device=torch.device('cuda')

#import XLMR and tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer=AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")
model=AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-large")

data=pd.read_csv('data/bongo_scam_cleaned.csv')


#Plot the lengths to determine padding for the tokenizer
sms=list(data['Sms'])
string_lengths=[len(s) for s in sms]
pd.Series(string_lengths).hist(bins=30)

"""The last bin seems to have significant frequency, hence it makes sense to select the maximum value"""

max_len=max(string_lengths)
print(max_len)

data['Category'].replace({'trust':0,'scam':1},inplace=True)

x=data['Sms']
y=data['Category']

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train=x_train.astype(str).tolist()
x_test=x_test.astype(str).tolist()

#encode the data and return it as pytorch tensors
x_train=tokenizer.batch_encode_plus(x_train,max_length=max_len,padding="max_length",return_token_type_ids=False,return_tensors="pt")
x_test=tokenizer.batch_encode_plus(x_test,max_length=max_len,padding="max_length",return_token_type_ids=False,return_tensors="pt")

#visualize the tokenized data by viewing 1 entry
x_train[0]

#covert target data to a tensor
train_y=torch.tensor(y_train.tolist(),dtype=torch.long)
test_y=torch.tensor(y_test.tolist(),dtype=torch.long)

#creating data loaders
training_data=TensorDataset(x_train['input_ids'],x_train['attention_mask'],train_y)
testing_data=TensorDataset(x_test['input_ids'],x_test['attention_mask'],test_y)

#sampler to be used during training
training_sampler=RandomSampler(training_data)
testing_sampler=RandomSampler(testing_data)

#define the batch size
batch_size=30
#data loader for training data
train_dataloader=DataLoader(training_data,sampler=training_sampler,batch_size=batch_size)

#data loader fot test data
test_dataloader=DataLoader(testing_data,sampler=testing_sampler,batch_size=batch_size)

"""Freezing the transformer parameters"""

for parameter in model.parameters():
  parameter.requires_grad=False

print(np.unique(train_y))

train_y.shape

train_y=np.array(train_y)
class_wts=compute_class_weight(class_weight='balanced',classes=np.unique(train_y),y=train_y)
print(class_wts)

#class weights to tensor
weights=torch.tensor(class_wts,dtype=torch.float)
weights=weights.to(device)

#loss function
cross_entropy=nn.NLLLoss(weight=weights)

#no. of training epochs
epochs=20.

"""Building the model structure"""

class XLMR(nn.Module):
  def __init__(self,model,num_classes=2):
    super(XLMR,self).__init__()
    self.transformer=model
    self.dropout=nn.Dropout(0.2)
    self.relu=nn.ReLU()
    self.fc1=nn.Linear(1024,512)
    self.fc2=nn.Linear(512,2)
    self.softmax=nn.LogSoftmax(dim=1)
  
  def forward(self,input_ids,attention_mask):
    output=self.transformer(input_ids=input_ids,attention_mask=attention_mask)
    pooled_output=output.last_hidden_state[:,0,:]
    x=self.fc1(pooled_output)
    x=self.relu(x)
    x=self.dropout(x)
    x=self.fc2(x)
    x=self.softmax(x)
    return x

f_model=XLMR(model)

f_model=f_model.to(device)

#optimizer
optimizer=AdamW(f_model.parameters(),lr=4e-3)

"""Finetuning"""

#training function
def train():
  f_model.train()
  total_loss=0
  predictions=[]
  for step,(id,mask,label) in enumerate(train_dataloader):
    if step%50==0 and not step==0:
      print('  Batch{:>5,}  of  {:>5,}.'.format(step,len(train_dataloader)))
    id,mask,label=id.to(device),mask.to(device),label.to(device)
    f_model.zero_grad()
    preds=f_model(id,mask)
    predictions.append(preds)
    loss=cross_entropy(preds,label)
    total_loss+=loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(f_model.parameters(),1.0)
    optimizer.step()
    preds=preds.detach().cpu().numpy()
  total_preds=np.concatenate(predictions,axis=0)
  mean_loss=total_loss/len(train_dataloader)
  return mean_loss,total_preds

#testing function
def test():
  f_model.eval()
  total_loss=0
  predictions=[]
  for step,(id,mask,label) in enumerate(test_dataloader):
    if step%50==0 and not step==0:
      print('  Batch{:>5,}  of  {:>5,}.'.format(step,len(train_dataloader)))
    id,mask,label=id.to(device),mask.to(device),label.to(device)
    with torch.no_grad():
      preds=f_model(id,mask)
      predictions.append(preds)
      loss=cross_entropy(preds,label)
      total_loss+=loss.item()
  total_preds=np.concatenate(predictions,axis=0)
  mean_loss=total_loss/len(train_dataloader)
  return mean_loss,total_preds

"""Training"""

train_losses=[]
test_losses=[]
best_loss=float('inf')

for epoch in range(int(epochs)):
  print("epoch",int(epoch)+1)
  train_loss,predictions=train()
  test_loss,predictions=test()
  train_losses.append(train_loss)
  test_losses.append(test_loss)
  print("training loss: {}".format(train_loss))
  print("test loss: {}".format(test_loss))

torch.save(f_model.state_dict(),'saved_weights.pt')

with torch.no_grad():
  test_seq,test_mask,_=next(iter(test_dataloader))
  test_seq,test_mask=test_seq.to(device),test_mask.to(device)
  preds=f_model(test_seq,test_mask)
  preds=preds.detach().cpu().numpy()