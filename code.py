# Import necessary libraries
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

# Preprocessing Function
def preprocess_text(text):
  # Lowercasing
  text = text.lower()
  # Remove HTML Tags
  text = re.sub(r"<.*?>"," ", text)
  # Remove URLs
  text = re.sub(r"https?://\S+|www\.\S+", " ", text)
  # Handle word contractions
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"\bain\'t\b", "is not", text)
  text = re.sub(r"\b(i|we)\'m\b", r"\1 am", text)
  # Remove non-word characters
  text = re.sub(r"\W", " ", text)
  # Remove extra whitespaces
  text = re.sub(r"\s+", " ", text)
  # Strip leading and trailing spaces
  text = text.strip()
  return text

# Read the dataset.csv file using pandas
dataset = pd.read_csv('IMDB Dataset.csv', index_col = False)

# Apply preprocessing stage in the dataset reviews
dataset['review'] = dataset['review'].apply(preprocess_text)

# Apply word tokenization in the dataset
dataset['tokens'] = dataset['review'].apply(word_tokenize)

# Change dataset labels to numbers
dataset['sentiment'] = dataset['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Divide the dataset into 80% training and 20% testing (for word2vec)
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(dataset['tokens'], dataset['sentiment'], test_size = 0.2, random_state = 1, stratify = dataset['sentiment'])

# Divide the dataset into 80% training and 20% testing (for bag of words and TFIDF)
X_train, X_test, y_train, y_test = train_test_split(dataset['review'], dataset['sentiment'], test_size = 0.2, random_state = 1, stratify = dataset['sentiment'])

# Using bag of words
cv = CountVectorizer()

# Fit the training data and transform the testing data
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()

# Use random forest classifier to train the model
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)

# Test the performance of the trained model in the test data
y_pred_rf = rf.predict(X_test_bow)
accuracy_score(y_test, y_pred_rf)

# Using bag of words but using bigrams
cv = CountVectorizer(ngram_range = (2,2))

# Fit the training data and transform the testing data
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()

# Use random forest classifier to train the model
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)

# Test the performance of the trained model in the test data
y_pred_rf = rf.predict(X_test_bow)
accuracy_score(y_test, y_pred_rf)

# Using TFIDF vectorizer
tfidf = TfidfVectorizer()

# Fit the training data and transform the testing data
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Use random forest classifier to train the model
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)

# Test the performance of the trained model in the test data
y_pred_rf = rf.predict(X_test_tfidf)
accuracy_score(y_test, y_pred_rf)

# Using word to vector model
w2v = Word2Vec(sentences = X_train_w2v)

# Define a function to convert sentence into vector
def sentence_vector(tokens, model, vector_size = 100):
  vec = [model.wv[word] for word in tokens if word in model.wv]
  if len(vec) == 0:
    return [0]*vector_size
  return np.mean(vec, axis = 0)

# Convert train and test data into vectors
X_train_w2v_vec = [sentence_vector(tokens, w2v) for tokens in X_train_w2v]
X_test_w2v_vec = [sentence_vector(tokens, w2v) for tokens in X_test_w2v]

# Use random forest classifier to train the model
rf = RandomForestClassifier()
rf.fit(X_train_w2v_vec, y_train_w2v)

# Test the performance of the trained model in the test data
y_pred_rf = rf.predict(X_test_w2v_vec)
accuracy_score(y_test_w2v, y_pred_rf)