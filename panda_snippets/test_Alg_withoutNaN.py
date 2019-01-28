#			TEST that is Algorithm is not working with NaN

import numpy
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dataset = read_csv('pima-indians-diabetes.data.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# split dataset into inputs and outputs

##############################
# drop rows with missing values
dataset.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(dataset.shape)
##############################



###############################################################################

values = dataset.values		#	WOW!!!! THIS IS RETURNS NUMPY INSTEAD OF PANDAS

###############################################################################

# print(dataset)
# print(type(dataset))

# print('\n\t\tGO')
# print(values)
# print(type(values))
# print(len(values))

X = values[:,0:8]
y = values[:,8]

# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(result.mean())