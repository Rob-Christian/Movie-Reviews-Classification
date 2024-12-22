# Movie Review Sentiment Analysis
This mini-project aims to classify movie reviews as either positive or negative using a Random Forest classifier. The goal is to demonstrate knowledge of data processing and model evaluation techniques for unstructured text inputs, without the use of transformer-based models.
In this project, we compare the performance of the Random Forest classifier across different data preprocessing methods:
1. Bag of Words (BOW) through Unigrams and Bigrams
2. Term Frequency-Inverse Document Frequency (TF-IDF)
3. Word-to-Vector (Word2Vec)
# Introduction
In this project, we classify movie reviews as positive or negative based on their text. We avoid using transformer-based models (such as BERT or GPT) to focus on understanding and evaluating traditional methods of text classification using a Random Forest classifier. The data is processed using different text vectorization techniques, and the accuracy of the classifier is compared to determine which preprocessing method provides the best results.
# Dataset
This project uses a dataset of movie reviews with corresponding sentiment labels. The dataset should be in a CSV file format (IMDB Dataset.csv) containing at least two columns:
1. review: texts of the movie review
2. sentiment: either positive or negative
You can download the dataset [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
# Preprocessing Steps
The following preprocessing steps are performed on the movie reviews:
1. Lowercasing: converts all text to lowercase for uniformity
2. HTML Tag Removal: Removes any HTML tags present in the text.
3. URL Removal: Removes URLs from the reviews.
4. Handling Contractions: Expands common contractions (e.g., won't to will not).
5. Removing Non-Word Characters: Strips out non-alphanumeric characters.
6. Whitespace Removal: Trims extra spaces.
# Vectorization Techniques
Different vectorization techniques are used to convert the text into numerical features that the Random Forest classifier can process:
1. Bag of Words (BoW): A simple model that treats each word as a feature. We experiment with both unigrams (individual words) and bigrams (pairs of consecutive words).
2. Term Frequency-Inverse Document Frequency (TF-IDF): This method weighs the importance of words in each review relative to their occurrence across the entire dataset, down-weighting common words and highlighting distinctive ones.
3. Word-to-Vector (Word2Vec): This technique learns the relationships between words based on their context. The Word2Vec model creates dense vector representations of words that capture semantic meaning, allowing us to represent the entire review as a vector.
# Model Training and Evaluation
A Random Forest classifier is used for training the model. Random Forest is an ensemble learning method that creates multiple decision trees and aggregates their results for classification. After training the model, we evaluate its performance on the testing set using the accuracy score. The accuracy is computed for each vectorization method
