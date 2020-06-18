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

