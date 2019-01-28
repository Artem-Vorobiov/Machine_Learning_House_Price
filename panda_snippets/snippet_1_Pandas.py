import pandas as pd

#		Print out name of Column and amount of Nan

df = pd.read_csv("data/train.csv")
count = 0

for column in df[:10]:
    if df[column].isnull().any():
       print('{} has {} null values'.format(column, df[column].isnull().sum()))
       count += 1

print('\n')
print('Amount of columns with NaN: ', count)

print('\n')
print('Amount of columns without NaN: ', (df.shape[1] - count))