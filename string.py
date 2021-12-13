import os
import pickle
import random
import numpy as numpy
import pefile
from sklearn import metrics
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


def pecheck(filename):
    f = open(filename, "rb")
    bytes = f.read(2)
    return bytes == b'MZ'

def get_string_features(filename, min_length,hasher):
    string_features = {}
    strings = os.popen("strings '{0}' ".format(filename)).read()
    strings = list(set(strings.split("\n")))

    for str in strings:
        if len(str) >= min_length:
            string_features[str] = 1

    # return string_features

    hashed_features = hasher.transform([string_features])
    hashed_features = hashed_features.todense()
    hashed_features = numpy.asarray(hashed_features)
    hashed_features = hashed_features[0]

    return hashed_features

hasher = FeatureHasher(n_features=200)

benignware_paths = []
for root, dirs, paths in os.walk('./data/benignware'):
    for path in paths:
        full_path = os.path.join(root, path)
        if pecheck(full_path):
            benignware_paths.append(full_path)

malware_paths = []
for root, dirs, paths in os.walk('./data/malware'):
    for path in paths:
        full_path = os.path.join(root, path)
        if pecheck(full_path):
            malware_paths.append(full_path)


x = [get_string_features(filename, 3, hasher) for filename in malware_paths + benignware_paths]
y = [1 for i in range(len(malware_paths))] + [0 for i in range(len(benignware_paths))]

x,y = numpy.array(x), numpy.array(y)
indicies = [i for i in range(len(y))]
random.shuffle(indicies)
x,y = x[indicies], y[indicies]

f1_score_1, f1_score_2, f1_score_3 = [], [], []
kfold = KFold(n_splits=10)
for indicies_training, indicies_test in kfold.split(indicies):
    training_x, training_y = x[list(indicies_training)], y[list(indicies_training)]
    test_x, test_y = x[list(indicies_test)], y[list(indicies_test)]

    classifier = RandomForestClassifier()
    classifier.fit(training_x, training_y)
    scores = classifier.predict_proba(test_x)[:, -1]
    y_pred = [0 if s < 0.4 else 1 for s in scores]
    f1_score_1.append(metrics.f1_score(test_y, y_pred))

    classifier1 = DecisionTreeClassifier()
    classifier1.fit(training_x, training_y)
    scores1 = classifier1.predict_proba(test_x)[:, -1]
    y_pred1 = [0 if s < 0.45 else 1 for s in scores1]
    f1_score_2.append(metrics.f1_score(test_y, y_pred1))

    classifier2 = LogisticRegression(max_iter=500)
    classifier2.fit(training_x, training_y)
    scores2 = classifier2.predict_proba(test_x)[:, -1]
    y_pred2 = [0 if s < 0.5 else 1 for s in scores2]
    f1_score_3.append(metrics.f1_score(test_y, y_pred2))


print(sum(f1_score_1)/len(f1_score_1))
print(sum(f1_score_2)/len(f1_score_2))
print(sum(f1_score_3)/len(f1_score_3))