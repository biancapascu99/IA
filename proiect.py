import codecs
import csv

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

file_train = codecs.open('date/train_samples.txt', encoding='utf-8')
file_test = codecs.open('date/test_samples.txt', encoding='utf-8')

test_samples = np.genfromtxt(file_test, delimiter='\t', dtype=None, names=('Id', 'Text'))
train_samples = np.genfromtxt(file_train, delimiter='\t', dtype=None, names=('Id', 'Text'))
train_labels = np.genfromtxt('date/train_labels.txt', delimiter='\t', names=('Id', 'Dial'))

# print(train_samples['Text'])

from nltk.tokenize import RegexpTokenizer

# for train in train_samples['Text']:
#     print(train)

# tokenizer = RegexpTokenizer(r'\w+')
# lista_cuvinte = []
# lista_index = []
# for train in train_samples['Text']:
#     lista_cuvinte.append(tokenizer.tokenize(train))
# for id in train_samples['Id']:
#     lista_index.append(id)

# for i in lista_cuvinte:
#     print(i)

sw = stopwords.words('romanian')
np.array(sw)

# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("romanian")
# fit the vectorizer using the text data
tfid_vectorizer.fit(train_samples['Text'])
# collect the vocabulary items used in the vectorizer
dictionary = tfid_vectorizer.vocabulary_.items()

fisier = csv.writer(open("dictionar.csv", "w", encoding='utf-8'))
for d in dictionary:
    fisier.writerow([d])
