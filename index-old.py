# library yang dibutuhkan
import pandas as pd  # Library untuk data analisa
import numpy as np   # Untuk arrays
import sklearn       # Untuk data manipulation
# class yang bakal dipakai
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# lokasi dataset malware yang bakal digunakan
malData = pd.read_csv("C:\\Users\\USER\\Desktop\\machine-learning-skripsi\\malware_dataset.csv", sep="|")

# memisahkan data menjadi malicious dan legitimate
legit = malData[0:41323].drop(["legitimate"], axis=1)
mal = malData[41323::].drop(["legitimate"], axis=1)

# tampilkan semua kolom
# pd.set_option("display.max_columns", None)

# Prepare the input data and labels for machine learning
data_in = malData.drop(['Name', 'md5', 'legitimate'], axis=1).values  # hide sementara kolom yang tidak dibutuhkan agar mempermudah proses data
labels = malData['legitimate'].values  # menyimpan kolom 'legitimate' menjadi variabel labels 

# masukkan data_in dan labels kedalam model ExtraTreesClassifier
extraTrees = ExtraTreesClassifier().fit(data_in, labels)

# simpan data-data yang penting menggunakan select dan nantinya data tersebut disimpan kedalam data_in_new
# parameter prefit=true berarti akan menggunakan transform() untuk mengolah data  
select = SelectFromModel(extraTrees, prefit=True)
data_in_new = select.transform(data_in)

# Get the number of selected features and their importances
features = data_in_new.shape[1]
importances = extraTrees.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Split the data into training and testing sets
legit_train, legit_test, mal_train, mal_test = train_test_split(data_in_new, labels, test_size=0.2)

# Create a RandomForestClassifier model and train it
classif = RandomForestClassifier(n_estimators=50)
classif.fit(legit_train, mal_train)

# Predict and evaluate false positives and false negatives
results = classif.predict(legit_test)
conf_mat = confusion_matrix(mal_test, results)

# Calculate and print false positives and false negatives as percentages
print("False positives: ", conf_mat[0][1] / sum(conf_mat)[0] * 100)
print("False negatives: ", conf_mat[1][0] / sum(conf_mat)[1] * 100)

# Calculate and print the accuracy of the machine learning algorithm
print("Algorithm score: ", classif.score(legit_test, mal_test) * 100)
