import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

malData = pd.read_csv("C:\\Users\\USER\\Desktop\\machine-learning-skripsi\\malware_dataset.csv", sep="|")

legit = malData[0:41323].drop("legitimate", axis=1)
mal = malData[41323::].drop("legitimate", axis=1)

data_in = malData.drop(['Name', 'md5', 'legitimate'], axis=1).values
labels = malData['legitimate'].values

extraTrees = ExtraTreesClassifier().fit(data_in, labels)

select = SelectFromModel(extraTrees, prefit=True)
data_in_new = select.transform(data_in)

legit_train, legit_test, mal_train, mal_test = train_test_split(data_in_new, labels, test_size=0.2)

classif = RandomForestClassifier(n_estimators=50)
classif.fit(legit_train, mal_train)

results = classif.predict(legit_test)
conf_mat = confusion_matrix(mal_test, results)

false_positives = conf_mat[0][1] / sum(conf_mat)[0] * 100
false_negatives = conf_mat[1][0] / sum(conf_mat)[1] * 100

accuracy = classif.score(legit_test, mal_test) * 100

print("False positives:", false_positives)
print("False negatives:", false_negatives)
print("Algorithm score:", accuracy)
