import pandas as pd
import numpy as np

test    = pd.read_csv('data/output_own.csv')
reality = pd.read_csv('data/sample_submission.csv')


# results = pd.DataFrame(data = [[reality],[test]], columns=['Sub', 'Test'])
result = pd.merge(test,reality,on='Id')
print(result)

count = 0
for index, row in result.iterrows():
    subtraction = abs(row['SalePrice_x'] - row['SalePrice_y'])
    count += 1
    accuracy = 1 - (subtraction/row['SalePrice_y'])
    # print(row['SalePrice_y'])
    # print(abs(subtraction))
    print(accuracy)
    print('\n\n\t\t TURN {}'.format(count))

print(count)

# COunt mean accuracy