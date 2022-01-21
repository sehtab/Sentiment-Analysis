#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 01:25:19 2021

@author: altair
"""

import pandas as pd
df = pd.read_csv('Tweets.csv')
#df = df.applymap(str)
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
import pandas as pd

# data analysis
df.airline.value_counts().plot(kind='pie', autopct='%1.0f%%', radius=2)
plt.show()
airline_sentiment = df.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')
plt.show()

import seaborn as sn
sn.barplot(x='airline_sentiment',y='airline_sentiment_confidence', data=df)
plt.show()

features = df.iloc[:, 10].values
labels = df.iloc[:,1].values

processed_features = []
for sentence in range(0, len(features)):
    # remove all the special characters
    processed_feature = re.sub(r'\W', '', str(features[sentence]))
    
    #remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+',' ', processed_feature)
    
    # remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    
    # substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
    # removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', ' ', processed_feature)
    
    # converting to lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)
    
# bag of words
vocab = ['I', 'like', 'to', 'play', 'football', 'it', 'is', 'a', 'good', 'game', 'prefer', 'over', 'rugby']

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500, min_df=7,max_df=0.8, stop_words=stopwords.words('english')) 
processed_features = vectorizer.fit_transform(processed_features).toarray()   

# dividing data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

#training the model
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(x_train, y_train)

#predictions
predictions = text_classifier.predict(x_test) 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Confusion Matrix:")
print("\n",confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Acuracy Score:",accuracy_score(y_test, predictions))