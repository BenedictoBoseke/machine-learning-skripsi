# link = https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/

# Evaluate using Shuffle Split Cross Validation
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import time
start_time = time.time()
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 42
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print("Process finished --- %s seconds ---" % (time.time() - start_time))