import pandas as pd
import numpy as np
import statistics 

test    = pd.read_csv('data/jan30_10col_AdaBoostClassifier_acc0136.csv')
reality = pd.read_csv('data/sample_submission.csv')


# results = pd.DataFrame(data = [[reality],[test]], columns=['Sub', 'Test'])
result = pd.merge(test,reality,on='Id')
# print(result)

count = 0
attemps =[]
for index, row in result.iterrows():
    subtraction = abs(row['SalePrice_x'] - row['SalePrice_y'])
    count += 1
    accuracy = 1 - (subtraction/row['SalePrice_y'])
    attemps.append(accuracy)
    # print(row['SalePrice_y'])
    # print(abs(subtraction))
    # print(accuracy)
    # print('\n\n\t\t TURN {}'.format(count))

# print(count)
# print(sum(attemps))
# print(sum(attemps)/count)				# One way
print(statistics.mean(attemps))		# Another way


# Results
# 1. Fitting Score 	= 0.0136
# 2. Predicted proces = [140000 140000 180000 ... 140000 180000 180000]
# 3. Comparison 		= 0.8488







