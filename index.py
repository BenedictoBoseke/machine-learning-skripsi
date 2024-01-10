import pandas as pd
import numpy as np
import time
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
start_time = time.time()

np.random.seed(42)

malData = pd.read_csv("malware_dataset.csv", sep="|")

legit = malData.iloc[0:41323].drop(["legitimate"], axis=1)
mal = malData.iloc[41323:].drop(["legitimate"], axis=1)

data_in = malData.drop(['Name', 'md5', 'legitimate'], axis=1).values
labels = malData['legitimate'].values

extraTrees = ExtraTreesClassifier().fit(data_in, labels)

select = SelectFromModel(extraTrees, prefit=True)
data_in_new = select.transform(data_in)

legit_train, legit_test, mal_train, mal_test = train_test_split(data_in_new, labels, test_size=0.2)

# Random Forest Classifier
classif_rf = RandomForestClassifier(n_estimators=50)
classif_rf.fit(legit_train, mal_train)

results_rf = classif_rf.predict(legit_test)
conf_mat_rf = confusion_matrix(mal_test, results_rf)

print("Random Forest Classifier:")
print("False positives: ", conf_mat_rf[0][1] / sum(conf_mat_rf)[0] * 100)
print("False negatives: ", conf_mat_rf[1][0] / sum(conf_mat_rf)[1] * 100)
print("Algorithm score: ", classif_rf.score(legit_test, mal_test) * 100)

# Gradient Boosting Classifier
classif_gb = GradientBoostingClassifier(n_estimators=50)
classif_gb.fit(legit_train, mal_train)

results_gb = classif_gb.predict(legit_test)
conf_mat_gb = confusion_matrix(mal_test, results_gb)

print("\nGradient Boosting Classifier:")
print("False positives: ", conf_mat_gb[0][1] / sum(conf_mat_gb)[0] * 100)
print("False negatives: ", conf_mat_gb[1][0] / sum(conf_mat_gb)[1] * 100)
print("Algorithm score: ", classif_gb.score(legit_test, mal_test) * 100)

# K-Nearest Neighbors (KNN) Classifier
classif_knn = KNeighborsClassifier(n_neighbors=5)
classif_knn.fit(legit_train, mal_train)

results_knn = classif_knn.predict(legit_test)
conf_mat_knn = confusion_matrix(mal_test, results_knn)

print("\nK-Nearest Neighbors (KNN) Classifier:")
print("False positives: ", conf_mat_knn[0][1] / sum(conf_mat_knn)[0] * 100)
print("False negatives: ", conf_mat_knn[1][0] / sum(conf_mat_knn)[1] * 100)
print("Algorithm score: ", classif_knn.score(legit_test, mal_test) * 100)

print("Process finished --- %s seconds ---" % (time.time() - start_time))