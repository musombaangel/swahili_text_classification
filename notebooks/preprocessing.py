import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

"""Loading the data"""

data=pd.read_csv("data/bongo_scam.csv")

"""Understanding the structure of the data and EDA"""

#preview the first 6 rows of the data
print(data.head(6))

#shape of the dataset
print("data shape: ",data.shape)

#check for balance in target variable
print("Count for each category: ",data['Category'].value_counts())

#The dataset has 2 columns and 1508 rows but has siignificantly more scam data than trustworthy data

#detect if there are any null values
print("null values",data.isnull().sum())
#No null values exist

#data types of the two columns
print("Data types",data.dtypes)

"""Preprocessing"""

#convert sms to a string to use string methods
data['Sms']=data['Sms'].astype(str)

#lowercase the sms column
data['Sms']=data['Sms'].apply(lambda x:x.lower())

#remove any special characters
data['Sms']=data['Sms'].apply(lambda txt:re.sub(r'\W|\d',' ',txt))

#stop word removal using a set of predefined Swahili stop words
stop_words=["akasema","hii","alikuwa","alisema","baada","basi","bila","cha","chini","hadi","hapo","hata","hivyo","hiyo","huku","huo","ili","ilikuwa","juu","kama","karibu","katika","kila","kima","kisha","kubwa","kutoka","kuwa","kwa","kwamba","kwenda","kwenye","la","lakini","mara","mdogo","mimi","mkubwa","mmoja","moja","muda","mwenye","na","naye","ndani","ng","ni","nini","nonkungu","pamoja","pia","sana","sasa","sauti","tafadhali","tena","tu","vile","wa","wakati","wake","walikuwa","wao","watu","wengine","wote","ya","yake","yangu","yao","yeye","yule","za","zaidi","zake"]
data['Sms']=data['Sms'].apply(lambda txt:' '.join([word for word in txt.split() if word not in (stop_words)]))


df=data
#run to save the clean dataset
df.to_csv('data/bongo_scam_cleaned.csv',index=False)