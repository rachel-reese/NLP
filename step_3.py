from gensim.models import Doc2Vec
from tensorflow import keras
import nltk
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import multiprocessing
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

# step 3
# create random forest and logistic regression models to predict bias
# print accuracy scores
# display confusion matricies

nltk.download('punkt')

dataset = pd.read_csv('./results.csv', header=0)
dataset = dataset[(dataset['bias'] == 'L') | (
    dataset['bias'] == 'R') | (dataset['bias'] == 'N')]
dataset = dataset.iloc[np.random.permutation(len(dataset))]

dataset['bias'] = dataset['bias'].replace(['L', 'N', 'R'], [0, 1, 2])


def clean(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = text.replace('„', '')
    text = text.replace('“', '')
    text = text.replace('"', '')
    text = text.replace('\'', '')
    text = text.replace('-', '')
    text = text.lower()
    return text


train, test = train_test_split(dataset, test_size=0.2)


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 3:
                continue
            tokens.append(word.lower())
    return tokens


train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['tweet']), tags=[r.bias]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['tweet']), tags=[r.bias]), axis=1)

cores = multiprocessing.cpu_count()
model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0,
                sample=0, min_count=2, workers=cores)


model.build_vocab(train_tagged.values)
model.train(utils.shuffle(train_tagged.values),
            total_examples=len(train_tagged.values), epochs=30)

model.save("doc2vec_tweets.model")


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    classes, features = zip(*[(doc.tags[0],
                               model.infer_vector(doc.words)) for doc in sents])
    return features, classes


train_x_0, train_y_0 = vec_for_learning(model, train_tagged)
test_x_0, test_y_0 = vec_for_learning(model, test_tagged)


def acc(true, pred):
    acc = 0
    for x, y in zip(true, pred):
        if (x == y):
            acc += 1
    return acc/len(pred)


# random forest
forest_0 = RandomForestClassifier(n_estimators=100)
forest_0.fit(train_x_0, train_y_0)
y_pred = forest_0.predict(test_x_0)
print(
    f'random forest accuracy score: {metrics.accuracy_score(test_y_0, forest_0.predict(test_x_0))}')

confusion_matrix = metrics.confusion_matrix(
    test_y_0, y_pred, labels=forest_0.classes_)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=forest_0.classes_)

cm_display.plot()
plt.show()

# logistic regression
scaler = StandardScaler()
X_train = scaler.fit_transform(train_x_0)
X_test = scaler.transform(test_x_0)

log_reg = LogisticRegression()
log_reg.fit(X_train, train_y_0)

print(
    f'logistic regression accuracy score: {metrics.accuracy_score(test_y_0, log_reg.predict(test_x_0))}')

confusion_matrix2 = metrics.confusion_matrix(
    test_y_0, y_pred, labels=log_reg.classes_)
cm_display2 = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix2, display_labels=log_reg.classes_)

cm_display2.plot()
plt.show()
