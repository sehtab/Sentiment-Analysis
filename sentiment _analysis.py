#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:42:01 2021

@author: altair
"""

import pandas as pd
df = pd.read_csv('Reviews.csv')
#df = df.applymap(str)
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df.hist('Score')
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Product Score')
plt.show()

# create some word clouds
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
#from nltk.corpus import WordCloud

# create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(['br','href'])
textt = ''.join(review for review in df.Text) 
wordcloud = WordCloud(stopwords=stopwords).generate(textt)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud11.png')
plt.show()

# classify tweets
# assign review with score > 3 as positive sentiment
# score < 3 negative sentiment
# avoid score = 3
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda rating: +1 if rating > 3 else -1)
print(df.head())

# split df - positive & negative sentiment:
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]
negative = negative.applymap(str)
# worldcloud - positive sentiment
stopwords = set(STOPWORDS)
stopwords.update(['br','href', "good", "great"])

# good and great removed coz they are included in negative comments
pos = ''.join(review for review in positive.Summary)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')
plt.show()  

# worldcloud - negativesentiment

neg = ",".join(review for review in negative.Summary)
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud33.png')
plt.show() 

df['sentimentt'] = df['sentiment'].replace({-1:'negative'})
df['sentimentt'] = df['sentimentt'].replace({1:'positive'})
print(df.head())
df.hist('sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Product Sentiment')
plt.show() 

# data cleaning
def remove_punctuation(text):
    final = "".join(u for u in text if u not in("?",".",";",":","!","'"))
    return final
df['Text']= df['Text'].apply(remove_punctuation)
df = df.dropna(subset=['Summary'])
df['Summary'] = df['Summary'].apply(remove_punctuation)

# split the dataframe
dfnew = df[['Summary','sentiment']]
print(dfnew.head()) 

# random split train and test data
index = df.index
df['random_number'] = np.random.randn(len(index))

train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]

# create bag of words
#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.fit_transform(test['Summary'])

# logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

x_train = train_matrix
x_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
#from sklearn.feature_extraction.text import TfidfVectorizer
#tvect = TfidfVectorizer(min_df=1, max_df=1)
#x_test=tvect.transform(x_test)
lr.fit(x_train, y_train)

# make predictions
from sklearn import *
##x_test = np.asarray(x_test)
from sklearn.feature_extraction.text import TfidfVectorizer
#tvect = TfidfVectorizer(min_df=1, max_df=1)
#x_test=tvect.transform(x_test)
predictions = lr.predict(x_test)

# find accuracy, precision, recall
from sklearn.metrics import confusion_matrix, classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)
print(classification_report(predictions, y_test))