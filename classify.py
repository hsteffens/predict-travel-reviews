from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('tripadvisor_review.csv')
dataset = dataset.drop('User ID', axis=1)

names = np.array(['Category 1','Category 2','Category 3','Category 4','Category 5','Category 6','Category 8'])

pd.options.display.float_format = '{:,.0f}'.format

dataset['Category 1'] = pd.to_numeric(dataset['Category 1'], downcast='integer')
dataset['Category 2'] = pd.to_numeric(dataset['Category 2'], downcast='integer')
dataset['Category 3'] = pd.to_numeric(dataset['Category 3'], downcast='integer')
dataset['Category 4'] = pd.to_numeric(dataset['Category 4'], downcast='integer')
dataset['Category 5'] = pd.to_numeric(dataset['Category 5'], downcast='integer')
dataset['Category 6'] = pd.to_numeric(dataset['Category 6'], downcast='integer')
dataset['Category 7'] = pd.to_numeric(dataset['Category 7'], downcast='integer')
dataset['Category 8'] = pd.to_numeric(dataset['Category 8'], downcast='integer')
dataset['Category 9'] = pd.to_numeric(dataset['Category 9'], downcast='integer')
dataset['Category 10'] = pd.to_numeric(dataset['Category 10'], downcast='integer')

n_samples = (len(dataset) * 0.50)

#Bootstrap split
random_state = 12883823
rkf = RepeatedKFold(n_splits=15, n_repeats=30, random_state=random_state)
result = next(rkf.split(dataset), None)

data_train = dataset.iloc[result[0]]
data_test = dataset.iloc[result[1]]

data = data_train[names]
target = data_train['Category 10'].astype('int')

scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)
data_test_transformed = scaler.transform(data_test[names])

# Create a classifier: a support vector classifier
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#classifier = svm.SVC(gamma='auto')
#classifier = RandomForestClassifier()
#classifier = MLPClassifier()

classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=11,
                       scoring='recall_macro')
classifier.fit(data_scaled, target)
expected = data_test['Category 10'].astype('int')

#predicted = cross_val_predict(classifier, data_test_transformed, expected, cv=9)
predicted = classifier.predict(data_test_transformed)

print(metrics.classification_report(expected.astype('int'), predicted.astype('int')))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected.astype('int'), predicted.astype('int')))
print(classifier.score(data_test[names].astype('int'), expected))





