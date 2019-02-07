from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
print('\n\t Loading data: \n', dataset, '\n\n')
print(dataset.shape)

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# print('\n\t\tInfo about X:\n', X)
# print('\n\t\tInfo about Y:\n', Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
confidence = model.score(X_test, y_test)
print('\n\t\t CONFIDENCE - 1\n')
print(confidence)
# print(model)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print('\n Model Predictions:\n')
print(y_pred)
print('\n Iterated over model:\n')
print(predictions)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))








