import codecs

# import csv
#
# import nltk
import csv
import pdb

import cv as cv
import nltk as nltk
import numpy as np
import unidecode

from nltk import SnowballStemmer
from sklearn import preprocessing, svm

# load data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

file_train = codecs.open('date/train_samples.txt', encoding='utf-8')
file_test = codecs.open('date/validation_target_samples.txt', encoding='utf-8')

test_data = np.genfromtxt(file_test, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_data = np.genfromtxt(file_train, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_labels = np.genfromtxt('date/train_labels.txt', delimiter='\t', names=('Id', 'Dial'))

training_data = training_data['Text']
training_labels = training_labels['Dial']
# test_labels = test_data['Id']
test_data = test_data['Text']

print(len(training_data))


def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler(with_mean=False)

    elif type == 'min_max':
        scaler = preprocessing.MaxAbsScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)


vector_prep = {'$NE$', 'de'}
# import nltk.stem.snowball._StandardStemmer

# ps = nltk.stem.snowball.RomanianStemmer

for k in range(len(training_data)):
    # training_data[k] = ps.stam(training_data[k])
    for prep in vector_prep:
    # training_data[k] = training_data[k].replace(" " + prep + ' ', ' ')
        training_data[k] = training_data[k].replace(prep, "")

print(training_data)
# unaccented_string=[unaccented_string]
tfid_vectorizer = TfidfVectorizer()
tfid_vectorizer.fit(training_data)
dictionary = tfid_vectorizer.vocabulary_.items()
print(len(dictionary))

fisier1 = csv.writer(open("dictionar.txt", "w", encoding='utf-8'))
for d in dictionary:
    fisier1.writerow([d])

print(vector_prep)

# fisier2 = csv.writer(open("dictionar_clear.txt", "w", encoding='utf-8'))
# print(len(dictionary))
# for d in dictionary:
#     fisier2.writerow(d)

train_features = tfid_vectorizer.transform(training_data)
test_features = tfid_vectorizer.transform(test_data)
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')
del training_data
del test_data
# summarize encoded vector
print(train_features.shape)
# print(type(vector))
# print(vector.toarray())

# nltk.stem.snowball.RomanianStemmer

# print(scaled_test_data)

svm_model = svm.SVC(C=0.55, kernel='linear')
svm_model.fit(scaled_train_data, training_labels)
predicted_labels_svm = svm_model.predict(scaled_test_data)

test_labels = np.genfromtxt('date/validation_target_labels.txt', dtype=None, delimiter='\t', names=('Id', 'Dial'))
print(accuracy_score(predicted_labels_svm, test_labels['Dial']))
print('f1 score', f1_score(np.asarray(test_labels['Dial']), predicted_labels_svm))
file = open("nou_cu_elim_cuv_linear.csv", "w", newline='')
fisier1 = csv.writer(file)
fisier1.writerow(['id'] + ['label'])

for id, pred in zip(test_labels['Id'], predicted_labels_svm):
    fisier1.writerow([str(id)] + [int(pred)])
    file.flush()

file.close()
#
# input_file = open("Predictii.csv","r+")
# reader_file = csv.reader(input_file)
# value = len(list(reader_file))
# print(value)

import nltk
# const stopwords = require('stopwords-ro');
# from nltk.corpus import stopwords
# set(stopwords.words('romania'))
#
# outfile = "cleaned_file.txt"
