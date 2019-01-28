import pandas as pd

#		For tests

dataset = pd.read_csv("data/train.csv", header=None)

print('\n\n')
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column
print(dataset.isnull().sum())
print('\n\n')
