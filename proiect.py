import codecs
import csv

import numpy as np
from sklearn import preprocessing, svm

# load data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score

file_train = codecs.open('date/train_samples.txt', encoding='utf-8')
file_test = codecs.open('date/validation_target_samples.txt', encoding='utf-8')

test_data = np.genfromtxt(file_test, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_data = np.genfromtxt(file_train, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_labels = np.genfromtxt('date/train_labels.txt', delimiter='\t', names=('Id', 'Dial'))

training_data = training_data['Text']
training_labels = training_labels['Dial']
test_data = test_data['Text']

print(len(training_data))


def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

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


from sklearn.feature_extraction.text import CountVectorizer

# list of text documents
# text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(training_data)
# summarize
print(vectorizer.vocabulary_)
# encode document

train_features = vectorizer.transform(training_data)
test_features = vectorizer.transform(test_data)
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')
del training_data
del test_data
# summarize encoded vector
print(train_features.shape)
# print(type(vector))
# print(vector.toarray())

# print(scaled_test_data)

svm_model = svm.SVC(C=0.63, kernel='linear')
svm_model.fit(scaled_train_data, training_labels)
predicted_labels_svm = svm_model.predict(scaled_test_data)

test_labels = np.genfromtxt('date/validation_target_labels.txt', dtype=None, delimiter='\t', names=('Id', 'Dial'))
print(accuracy_score(predicted_labels_svm, test_labels['Dial']))

file = open("Predictii.csv", "w")
fisier1 = csv.writer(file)
fisier1.writerow(['id', 'label'])
for id, pred in zip(test_labels['Id'], predicted_labels_svm):
    fisier1.writerow([str(id), int(pred)])
    file.flush()

file.close()
