import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
header=None,names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
train = df[test_idx==True]
test = df[test_idx==False]

features = ['sepal_length', 'sepal_width']

results = []
for n in range(1, 45, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[features], train['class'])
    preds = clf.predict(test[features])
    accuracy = np.where(preds==test['class'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()