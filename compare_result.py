import pandas as pd
import numpy as np
import statistics 

test    = pd.read_csv('data/jan31_full_RandomForestRegressor_acc9994.csv')
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



########################	RESULTS.    ########################
#				SVM

# 1.		Trained acc = 0.5
# Compare (acc) = 0.7942

# 2.		Trained acc = 0.75
# Compare (acc) = 0.7463 (0.7494) (0.7527)


#				DecisionTreeRegressor

# 3.		Trained acc = 0.6861 
# Compare (acc) = 0.6779; 

# 4.		Trained acc = 0.75
# Compare (acc) = 0.69


#				LinearRegression

# 5.		Trained acc = 0.66
# Compare (acc) = 0.7153

# 6.		Trained acc = 0.6888
# Compare (acc) = 0.7224


#				KNeighborsClassifier

# 7.		Trained acc = 0.0034
# Compare (acc) = 0.7066

# 8.		Trained acc = 0.0068
# Compare (acc) = 0.7046


#				LogisticRegression


# 9.		Trained acc = 0.01369
# Compare (acc) = 0.7103

# 10.		Trained acc = 0.0068
# Compare (acc) = 0.7028


#				MLPClassifier


# 11.		Trained acc = 0.0136
# Compare (acc) = 0.6873




#				DecisionTreeClassifier


# 12.		Trained acc = 0.0136
# Compare (acc) = 0.8049




#				RandomForestRegressor


# 13.		Trained acc = 0.7905
# Compare (acc) = 0.7084




#				GaussianNB


# 14.		Trained acc = 0.0102
# Compare (acc) = 0.7332


#				AdaBoostClassifier


# 15.		Trained acc = 0.0068
# Compare (acc) = 0.8542
# 16.		Trained acc = 0.0205
# Compare (acc) = 0.8488


# 				JANUARY 31

#				RandomForestRegressor

# Withfull feature set + scaler
# 17.		Trained acc = 0.8994
# Compare (acc) = 0.6817


# Withfull feature set + without scaler
# 18.		Trained acc = 0.9105
# Compare (acc) = 0.7003








