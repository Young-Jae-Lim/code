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

def String_PE_IAT(fullpath,min_length,hasher):
    IAT_features = {}
    try:
        pe = pefile.PE(fullpath)
        pe.parse_data_directories()
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                try:
                    c = "{0}".format(imp.name)
                    c = list(set(c.split("\n")))
                    for iat in c:
                        if len(iat) >= min_length:
                            IAT_features[iat] = 1
                except:
                    pass
    except:
        strings = os.popen("strings '{0}' ".format(fullpath)).read()
        strings = list(set(strings.split("\n")))

        for str in strings:
            if len(str) >= min_length:
                IAT_features[str] = 1

    hashed_features = hasher.transform([IAT_features])
    hashed_features = hashed_features.todense()
    hashed_features = numpy.asarray(hashed_features)
    hashed_features = hashed_features[0]
    return hashed_features

hasher = FeatureHasher(n_features=1000)

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

IAT_x = []
for file in malware_paths + benignware_paths:
    attributes = String_PE_IAT(file, 3, hasher)
    IAT_x.append(attributes)
IAT_x = numpy.array(IAT_x, dtype=object)

y = [1 for i in range(len(malware_paths))] + [0 for i in range(len(benignware_paths))]
y = numpy.array(y)
indicies = [i for i in range(len(y))]
random.shuffle(indicies)
IAT_x = IAT_x[indicies]
y = y[indicies]

f1_score_1, f1_score_2, f1_score_3 = [], [], []
kfold = KFold(n_splits=10)
for indicies_training, indicies_test in kfold.split(indicies):
    IAT_training_x, IAT_training_y = IAT_x[list(indicies_training)], y[list(indicies_training)]
    test_x, test_y = IAT_x[list(indicies_test)], y[list(indicies_test)]

    classifier = RandomForestClassifier()
    classifier.fit(IAT_training_x, IAT_training_y)
    scores = classifier.predict_proba(test_x)[:, -1]
    y_pred = [0 if s < 0.4 else 1 for s in scores]
    f1_score_1.append(metrics.f1_score(test_y, y_pred))

    classifier1 = DecisionTreeClassifier()
    classifier1.fit(IAT_training_x, IAT_training_y)
    scores1 = classifier1.predict_proba(test_x)[:, -1]
    y_pred1 = [0 if s < 0.45 else 1 for s in scores1]
    f1_score_2.append(metrics.f1_score(test_y, y_pred1))

    classifier2 = LogisticRegression(max_iter=500)
    classifier2.fit(IAT_training_x, IAT_training_y)
    scores2 = classifier2.predict_proba(test_x)[:, -1]
    y_pred2 = [0 if s < 0.5 else 1 for s in scores2]
    f1_score_3.append(metrics.f1_score(test_y, y_pred2))


print(sum(f1_score_1)/len(f1_score_1))
print(sum(f1_score_2)/len(f1_score_2))
print(sum(f1_score_3)/len(f1_score_3))