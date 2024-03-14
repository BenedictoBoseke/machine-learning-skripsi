import pandas as pd
import numpy as np
import time
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


malData = pd.read_csv("malware_dataset.csv", sep="|")

Data = malData.drop(['Name', 'md5', 'legitimate'], axis=1).values
Target = malData['legitimate'].values
FeatSelect = ExtraTreesClassifier().fit(Data, Target)
Model = SelectFromModel(FeatSelect, prefit=True)
Data_new = Model.transform(Data)

Features = Data_new.shape[1]
index = np.argsort(ExtraTreesClassifier().fit(Data, Target).feature_importances_)[::-1][:Features]

for feat in range(Features):
    print(malData.columns[2 + index[feat]])
