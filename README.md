# Sentiment-Analysis
# Problem Statement

In this project, tweet about six airlines are analyzed to predict a tweet contains positive, negative or neutral sentiment about the airline. This is a typical supervised learning task where given a text strring, we have to categorize the text sring into predefined categories.

# Project Outline

    Data Analysis
    Data Cleaning
    TF-IDF
    Making Predictions and Evaluating the Model

# Data Analysis

Data is analyzed with proper pie chart and bar chart. At last, sentiment confidence is plotted with respect to positive, negative and neutral comments.

# Data Cleaning

Tweets contain many slang words and punctuation marks. Those are cleaned before used for training the machine learning model.

# TF-IDF

Term frequency and inverse document frequency is used by TfidfVectorizer in Scikit-Learn to convert string to TF-IDF featured vector. The idea behind is the words that occur less in all the documents and more in individual document contribute more toward classification.
Making Predictions and Evaluating the Model

Prediction is done with RandomForsetClassifier class which is also used for training. From that classification_report, confusion_matrix and accuracy_score are computed. This model has accuracy score of 64%.
