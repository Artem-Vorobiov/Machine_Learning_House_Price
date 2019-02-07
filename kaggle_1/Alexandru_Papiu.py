# Source - https://www.kaggle.com/apapiu/regularized-linear-models
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import statistics 

# Step 1: Take train and test data, then concatenate them and I have one big Pandas DataFrame
# Step 2: Transform the skewed numeric features by taking log(feature + 1)
# Step 3: Create Dummy variables for the categorical features


#	Loading data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")



#	NEW WAY
# Preprocess train and test data in the very beginning simoltaneously
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
# print(all_data)
# print('\n\n\t\tALL. ', all_data.shape)
# print('\n\n\t\tTRAIN', train.shape)
# print('\n\n\t\tTEST ', test.shape)

# NOTE -> below you can see the way of choosing distance between cols
# print('\n\n\t\tTrain', train.loc[:,'MSSubClass':'SaleCondition'].shape)
# print('\n\n\t\tTest ', test.loc[:,'MSSubClass':'SaleCondition'].shape)



#	PLAN
# 1. First I'll transform the skewed numeric features by taking log(feature + 1) - 
#	 this will make the features more normal
# 2. Create Dummy variables for the categorical features
# 3. Replace the numeric missing values (NaN's) with the mean of their respective columns
#	
# numpy.log1p --> Return the natural logarithm of one plus the input array, element-wise.
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# print('\n\n')
# print(train["SalePrice"])
# print(type(train["SalePrice"]))
# print(train["SalePrice"].shape)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()
# plt.show()





#log transform the target:
# train["SalePrice"] = np.log1p(train["SalePrice"])				# Optional


#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# print(numeric_feats)
# print(train[numeric_feats])			#	[1460 rows x 36 columns]

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# print('\n\n\n')
# print(skewed_feats)

skewed_feats = skewed_feats[skewed_feats > 0.75] 					# Choose those that mpre than 0.75
skewed_feats = skewed_feats.index
# print('\n\n\n')
# print(skewed_feats)

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# print('\n\n\n')
# print(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
# print('\n\n\n')
# print(all_data)


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice






from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


###########################################################################################
###################		Ridge()		###################
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]







#####	Try out ONE
#################################################

# minn = min(cv_ridge)
# model_ridge = Ridge(minn).fit(X_train, y)


# zzz = pd.DataFrame(model_ridge.predict(X_test))
# actual_price = pd.read_csv('../data/sample_submission.csv')
# # actual_price["SalePrice"] = np.log1p(actual_price["SalePrice"])		# Optional
# all_data = pd.concat([actual_price, zzz], axis=1)
# all_data.drop("Id", axis = 1, inplace = True)
# all_data.columns = ['ActualPrice', 'PredictedPrice']
# print(all_data)
# print(all_data.shape)
# print(type(all_data))


# count = 0
# attemps =[]
# for index, row in all_data.iterrows():
#     subtraction = abs(row['ActualPrice'] - row['PredictedPrice'])
#     count += 1
#     accuracy = 1 - (subtraction/row['ActualPrice'])
#     attemps.append(accuracy)
# print(statistics.mean(attemps))		# Another way

# zzz = zzz.values
# range = np.arange(1461, 2920)
# with open('../data/AlexandruRidge.csv', 'w') as f:
#     f.write("Id,SalePrice\n")
#     for x, y in zip(range, zzz):
#         f.write("{}, {} \n".format(x, float(y)))
#     f.close()
#################################################





# cv_ridge = pd.Series(cv_ridge, index = alphas)
# # cv_ridge.plot(title = "Validation - Just Do It")
# # plt.xlabel("alpha")
# # plt.ylabel("rmse")
# # plt.show()

# cv_ridge.min()
# print(cv_ridge)
###########################################################################################








###########################################################################################
###################		LassoCV()		###################

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
print(rmse_cv(model_lasso).mean())

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
# print(coef)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
print(imp_coef)


#####	Try out ONE
#################################################
# zzz = pd.DataFrame(model_lasso.predict(X_test))
# actual_price = pd.read_csv('../data/sample_submission.csv')
# # actual_price["SalePrice"] = np.log1p(actual_price["SalePrice"])		# Optional
# all_data = pd.concat([actual_price, zzz], axis=1)
# all_data.drop("Id", axis = 1, inplace = True)
# all_data.columns = ['ActualPrice', 'PredictedPrice']
# print(all_data)
# print(all_data.shape)
# print(type(all_data))


# count = 0
# attemps =[]
# for index, row in all_data.iterrows():
#     subtraction = abs(row['ActualPrice'] - row['PredictedPrice'])
#     count += 1
#     accuracy = 1 - (subtraction/row['ActualPrice'])
#     attemps.append(accuracy)
# print(statistics.mean(attemps))		# Another way

# zzz = zzz.values
# range = np.arange(1461, 2920)
# with open('../data/AlexandruLasso.csv', 'w') as f:
#     f.write("Id,SalePrice\n")
#     for x, y in zip(range, zzz):
#         f.write("{}, {} \n".format(x, float(y)))
#     f.close()
#################################################


# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")
# plt.show()


#let's look at the residuals as well:
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
# preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")
# # plt.show()






###########################################################################################
###################		xgboost()		###################

# import xgboost as xgb

# dtrain = xgb.DMatrix(X_train, label = y)
# dtest = xgb.DMatrix(X_test)

# params = {"max_depth":2, "eta":0.1}
# model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

# model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
# model_xgb.fit(X_train, y)

# # xgb_preds   = np.expm1(model_xgb.predict(X_test))
# # lasso_preds = np.expm1(model_lasso.predict(X_test))

# # predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
# # predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
# #	ДОБАВИТЬ ТУТ ТРЕТЬЮ КРИВУЮ ВЗЯТУЮ ИЗ ОТВЕТОВ!!!!


# preds = 0.7*lasso_preds + 0.3*xgb_preds
# # solution = pd.DataFrame({"SalePrice":preds, "Id":test.Id})
# # solution.to_csv("ridge_sol.csv", index = False)

#####	Try out ONE
#################################################
# zzz = pd.DataFrame(model_xgb.predict(X_test))
# actual_price = pd.read_csv('../data/sample_submission.csv')
# # actual_price["SalePrice"] = np.log1p(actual_price["SalePrice"])		# Optional
# all_data = pd.concat([actual_price, zzz], axis=1)
# all_data.drop("Id", axis = 1, inplace = True)
# all_data.columns = ['ActualPrice', 'PredictedPrice']
# print(all_data)
# print(all_data.shape)
# print(type(all_data))


# count = 0
# attemps =[]
# for index, row in all_data.iterrows():
#     subtraction = abs(row['ActualPrice'] - row['PredictedPrice'])
#     count += 1
#     accuracy = 1 - (subtraction/row['ActualPrice'])
#     attemps.append(accuracy)
# print(statistics.mean(attemps))		# Another way

# zzz = zzz.values
# range = np.arange(1461, 2920)
# with open('../data/AlexandruXGB&Lasso.csv', 'w') as f:
#     f.write("Id,SalePrice\n")
#     for x, y in zip(range, zzz):
#         f.write("{}, {} \n".format(x, float(y)))
#     f.close()
#################################################








###########################################################################################
###################		Kernel()		###################
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.regularizers import l1
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # X_train = StandardScaler().fit_transform(X_train)
# X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
# print(X_tr.shape)



# model = Sequential()
# model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
# model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

# model.compile(loss = "mse", optimizer = "adam")
# model.summary()
# hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs = 25)
# www = pd.DataFrame(model.predict(X_val)[:,0])



#####	Try out ONE
#########################################
# # X_test = StandardScaler().fit_transform(X_test)
# zzz = pd.DataFrame(model.predict(X_test))
# actual_price = pd.read_csv('../data/sample_submission.csv')
# # actual_price["SalePrice"] = np.log1p(actual_price["SalePrice"])		# Optional
# all_data = pd.concat([actual_price, zzz], axis=1)
# all_data.drop("Id", axis = 1, inplace = True)
# all_data.columns = ['ActualPrice', 'PredictedPrice']
# print(all_data)
# print(all_data.shape)
# print(type(all_data))


# count = 0
# attemps =[]
# for index, row in all_data.iterrows():
#     subtraction = abs(row['ActualPrice'] - row['PredictedPrice'])
#     count += 1
#     accuracy = 1 - (subtraction/row['ActualPrice'])
#     attemps.append(accuracy)
# print(statistics.mean(attemps))		# Another way

# zzz = zzz.values
# range = np.arange(1461, 2920)
# with open('../data/Alexandru2.csv', 'w') as f:
#     f.write("Id,SalePrice\n")
#     for x, y in zip(range, zzz):
#         f.write("{}, {} \n".format(x, float(y)))
#     f.close()
#########################################





#####	Try out TWO
#########################################
# www = pd.DataFrame(model.predict(X_val)[:,0], columns = ['Predicted_Price'])
# www = www.reset_index()
# www.drop("index", axis = 1, inplace = True)
# zzz = y_val.to_frame(name='Actual_Price')
# zzz = zzz.reset_index()
# zzz.drop("index", axis = 1, inplace = True)
# print(www.shape)
# print(zzz.shape)
# all_data = pd.concat([zzz, www], axis=1)
# print('\n\n\n')
# print(all_data)
# all_data.hist()
# plt.show()
#########################################















































