import numpy as np
import re
import nltk
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
# import cupy as np

review_data = load_files(r"movie_review")
X, label = review_data.data, review_data.target

documents = []

stemmer = WordNetLemmatizer()

# for review in data:

#     review = review.lower()

#     document = re.sub(r'\W', ' ', str(review))

#     document = re.sub(r'^b\s+', '', document)

#     document = document.split()

#     document = [word for word in document if word.isalpha()]

#     document = [word for word in document if not word in list(stopwords.words('english'))]

#     document = [word for word in document if  len(word) > 2]

#     document = [stemmer.lemmatize(word) for word in document]

#     # document = [word[0] for word in pos_tag(document) if word[1] == 'JJ' or word[1] == 'NN' or word[1] == 'VBG']
#     # print(document)
#     # break


#     document = ' '.join(document)

#     documents.append(document)


for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.8)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, shuffle=False)

y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(z) * (1 - sigmoid(z))


def accuracy(label, sig):
    correct = 0
    label = label.T
    sig = sig.T
    for i in range(len(label)):
        # print(label[i][0])
        # print(sig[i])
        if label[i][0] == 1 and sig[i] >= 0.5:
            correct += 1

        if label[i][0] == 0 and sig[i] < 0.5:
            correct += 1

    return (correct / len(label)) * 100


n1 = 256
n2 = 32

lambda_value = 0
n = 1500 * n1 + n1 * n2 + n2 * 1
J_list = []
test_J_list = []
accuracy_list = []
test_accuracy_list = []
count = 0
flag=True

import math
u = np.random.randn(n1, 1500)
v = np.random.randn(n2, n1)
w = np.random.randn(1, n2)
# b1 = np.zeros((196,1))
# b2 = np.zeros((49,1))
# b3 = np.zeros((1,1))
b1 = np.zeros((n1,1))
b2 = np.zeros((n2,1))
b3 = np.zeros((1,1))

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

learning_rate = 0.1

for cnt in tqdm(range(150000)):
    J = 0
    test_J = 0
    y_ = np.dot(u, X_train) + b1
    y = sigmoid(y_)
    z_ = np.dot(v, y) + b2
    z = sigmoid(z_)
    h_ = np.dot(w, z) + b3
    h = sigmoid(h_)
    # calculate loss about train, test data
    J = (-1 / 1401) * (
                np.sum(((y_train) * np.log(h) + (1 - y_train) * np.log(1 - h))) + lambda_value * (1 / (2 * 600)) * (
                    np.sum(u * u) + np.sum(v * v) + np.sum(w * w)))
    J_list.append(J)
    test_y_ = np.dot(u, X_test) + b1
    test_y = sigmoid(test_y_)
    test_z_ = np.dot(v, test_y) + b2
    test_z = sigmoid(test_z_)
    test_h_ = np.dot(w, test_z) + b3
    test_h = sigmoid(test_h_)
    test_J = (-1 / 601) * (np.sum(((y_test) * np.log(test_h) + (1 - y_test) * np.log(1 - test_h))) + lambda_value * (
                1 / (2 * 600)) * (np.sum(u * u) + np.sum(v * v) + np.sum(w * w)))
    test_J_list.append(test_J)
    # calculate gradient descent and update

    df_h_ = h - y_train
    df_w = (1 / 1401) * (np.dot(df_h_, z.T) + (lambda_value / (2 * 600)) * w)
    w -= learning_rate * df_w
    df_z_ = np.dot(w.T, df_h_) * (z * (1 - z))
    df_v = (1 / 1401) * (np.dot(df_z_, y.T) + (lambda_value / (2 * 600)) * v)

    v -= learning_rate * df_v
    df_y_ = np.dot(v.T, df_z_) * (y * (1 - y))
    df_u = (1 / 1401) * (np.dot(df_y_, X_train.T) + (lambda_value / (2 * 600)) * u)
    u -= learning_rate * df_u

    df_b3 = (1 / 1401) * np.sum(df_h_, axis=1, keepdims=True)
    df_b2 = (1 / 1401) * np.sum(df_z_, axis=1, keepdims=True)
    df_b1 = (1 / 1401) * np.sum(df_y_, axis=1, keepdims=True)

    b3 -= learning_rate * df_b3
    b2 -= learning_rate * df_b2
    b1 -= learning_rate * df_b1
    # calculate the accuracy about train, test data
    accuracy_list.append(accuracy(y_train, h[0]))
    test_accuracy_list.append(accuracy(y_test, test_h[0]))
# print(accuracy(testing_label, test_h))

y_train_pred = []
for val in h.T:
  if val < 0.5:
    y_train_pred.append(0)
  else:
    y_train_pred.append(1)