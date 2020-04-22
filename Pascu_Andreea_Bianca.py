import codecs
import csv
import numpy as np
import sklearn
from sklearn import preprocessing, svm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import ComplementNB

file_train = codecs.open('train_samples.txt', encoding='utf-8')
file_test = codecs.open('validation_samples.txt', encoding='utf-8')

test_data = np.genfromtxt(file_test, comments=None, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_data = np.genfromtxt(file_train, comments=None, delimiter='\t', dtype=None, names=('Id', 'Text'))
training_labels = np.genfromtxt('train_labels.txt', delimiter='\t', names=('Id', 'Dial'))
test_labels = np.genfromtxt('validation_labels.txt', dtype=None, delimiter='\t', names=('Id', 'Dial'))

training_data = training_data['Text']
training_labels = training_labels['Dial']
test_id = test_data['Id']
test_data = test_data['Text']
test_labels = test_labels['Dial']


# functie pentru normalizarea datelor

def normalize_data(train_data, test_data, type=None):
    normalize = None

    if type == 'l1':
        normalize = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        normalize = preprocessing.Normalizer(norm='l2')

    if normalize is not None:
        normalize.fit(train_data)
        normalize_train_data = normalize.transform(train_data)
        normalize_test_data = normalize.transform(test_data)
        return (normalize_train_data, normalize_test_data)
    else:
        print("Nu s-a realizat nicio normalizare.")
        return (train_data, test_data)


# crearea unui vocabular
vocabulary = CountVectorizer(ngram_range=(1, 2), token_pattern="[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+")
vocabulary.fit(training_data)
dictionary = vocabulary.vocabulary_.items()

file_dictionary = csv.writer(open("dictionar.txt", "w", encoding='utf-8'))
for d in dictionary:
    file_dictionary.writerow([d])

# codificarea datelor
train_features = vocabulary.transform(training_data)
test_features = vocabulary.transform(test_data)

# normalizarea datelor de test si de train
normalize_train_data, normalize_test_data = normalize_data(train_features, test_features, type='l2')
del training_data
del test_data
print('SVM cu kernel rbf')
# # varianta cu SVM

# svm kernel rbf
svm_model = svm.SVC(C=1, kernel='rbf')

# antrenarea modelului
svm_model.fit(normalize_train_data, training_labels)
predicted_labels_rbf = svm_model.predict(normalize_test_data)
print('accuracy_score', sklearn.metrics.accuracy_score(predicted_labels_rbf, test_labels))
print('f1 score', f1_score(np.asarray(test_labels), predicted_labels_rbf))

print('---------------------')
print('SVM cu kernel linear')

# svm cu kernel linear
svm_model_linear = svm.LinearSVC(C=2)

# antrenarea modelului
svm_model_linear.fit(normalize_train_data, training_labels)
predicted_labels_linear = svm_model_linear.predict(normalize_test_data)
print('accuracy_score', sklearn.metrics.accuracy_score(predicted_labels_linear, test_labels))
print('f1 score', f1_score(np.asarray(test_labels), predicted_labels_linear))

print('---------------------')
print('CNB')

# varianta ci CNB
cnb = ComplementNB(alpha=0.027)

# antrenarea modelului
cnb.fit(normalize_train_data, training_labels)
predicted_labels_cnb = cnb.predict(normalize_test_data)
print('accuracy_score', sklearn.metrics.accuracy_score(predicted_labels_cnb, test_labels))
print('f1 score', f1_score(np.asarray(test_labels), predicted_labels_cnb))

# scriere in fisier pentru submit

file1 = open("sample_submission_rbf.csv", "w", newline='')
file_submission1 = csv.writer(file1)
file_submission1.writerow(['id'] + ['label'])

for id, label in zip(test_id, predicted_labels_rbf):
    file_submission1.writerow([str(id)] + [int(label)])
file1.flush()
file1.close()

file2 = open("sample_submission_linear.csv", "w", newline='')
file_submission2 = csv.writer(file2)
file_submission2.writerow(['id'] + ['label'])

for id, label in zip(test_id, predicted_labels_linear):
    file_submission2.writerow([str(id)] + [int(label)])
file2.flush()
file2.close()

file3 = open("sample_submission_cnb.csv", "w", newline='')
file_submission3 = csv.writer(file3)
file_submission3.writerow(['id'] + ['label'])

for id, label in zip(test_id, predicted_labels_cnb):
    file_submission3.writerow([str(id)] + [int(label)])
file3.flush()
file3.close()
