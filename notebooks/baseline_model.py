# Importing the necessary libraries
import nltk
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd

data=pd.read_csv("data/bongo_scam_cleaned.csv")

"""Baseline Model"""

#vectorizer
vectorizer=TfidfVectorizer()

x=data['Sms']
y=data['Category']

#splitting the data into training and testing sets
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=46)

x_train=vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

#set model and hyperparameters
model=LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)

#fitting the model
model.fit(x_train, y_train)

#testing the model
predictions=model.predict(x_test)


#evaluation
score=f1_score(y_test, predictions, average='weighted')
print("f1 score: ",score)

conf_matrix=confusion_matrix(y_test, predictions)
print("confusion matrix",conf_matrix)

#f1 score:  0.9824561403508771
#confusion matrix [[ 241   7]
# [ 1 128]]
