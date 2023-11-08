# data-analysis

Project Name: Sentiment Analysis of COVID-19 Tweets

Project Description:
This project aims to perform sentiment analysis on a dataset of tweets related to the COVID-19 pandemic. The project involves several key components:

1. Data Preparation:

The dataset is read from a CSV file named "Corona_NLP.csv" using the pandas library.
Text data within the "OriginalTweet" column is cleaned and preprocessed by removing mentions, hashtags, non-alphanumeric characters, and URLs using regular expressions.
The dataset is split into a training set and a testing set for model evaluation.
2. Text Preprocessing:

The Natural Language Toolkit (nltk) is used to download a set of English stopwords to improve text data quality.
The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is applied to the cleaned text data to convert it into numerical features for machine learning.
3. Sentiment Analysis Model:

A Multinomial Naive Bayes classifier is used for sentiment analysis. It's a supervised machine learning model that learns to classify text data into predefined sentiment categories.
Sentiment labels are encoded using the LabelEncoder from Scikit-Learn to convert them into numerical values for training and evaluation.
The model is trained on the TF-IDF features of the training data.
4. Testing and Prediction:

The same preprocessing steps used for the training data are applied to the testing data.
The trained Naive Bayes classifier is used to predict sentiment labels for the testing data.
5. Evaluation:

The performance of the sentiment analysis model is evaluated using an ROC (Receiver Operating Characteristic) curve, which plots the true positive rate against the false positive rate.
The Area Under the Curve (AUC) of the ROC curve is calculated as a measure of the model's predictive accuracy.
6. Data Visualization:

The Seaborn and Matplotlib libraries are used to create visualizations, including the ROC curve.
Technologies and Libraries Used:

Python
pandas
numpy
re (regular expressions)
nltk (Natural Language Toolkit)
Scikit-Learn (sklearn)
Matplotlib
Seaborn
