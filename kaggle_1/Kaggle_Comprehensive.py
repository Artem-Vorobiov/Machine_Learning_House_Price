#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 					#	Seaborn is a Python data visualization library based on matplotlib.
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
sns.set(style="darkgrid")

warnings.filterwarnings('ignore')
# %matplotlib inline

df_train = pd.read_csv('data/train.csv')
# print(df_train.columns)
# print(df_train.index)

#descriptive statistics summary
print('\n')
print(df_train['SalePrice'].describe())

# histogram
# print(sns.distplot(df_train['SalePrice']))
# sns.distplot(df_train['SalePrice'])


#skewness and kurtosis
print('\n')
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice
# print('\n')
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))



#scatter plot totalbsmtsf/saleprice
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



#box plot overallqual/saleprice
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);



# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(16, 8))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);
# plt.xticks(rotation=90)			# Turning side of typing for numbers that Hightlights the Graph



#correlation matrix
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)


#saleprice correlation matrix
# corrmat = df_train.corr()
# k = 10 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)



#scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size = 1.2)
# plt.show()





#missing data
#########################################################################################
#########################################################################################
print('\n')
print('\t\t\t missing data\n')
# www = df_train.isnull()										# Shows all the Pandas DataFrame witg True and False
# www = df_train.isnull().sum()									# Shows 2 columns: first is title('Fence'), second is sum  of NaN's ('1179')
# www = df_train.isnull().sum().sort_values(ascending=False)  	# Shows 2 columns: first is title('Fence'), second is sum  of NaN's ('1179')
																# However first value is one that has the large sum of NaN's
# www = df_train['SalePrice'].sort_values()						# sort_values() - shows price values from lowest to highest
# www = df_train['SalePrice'].sort_values(ascending=False)		# sort_values() - shows price values from highest to lowest
# print(www)

#	FIRST how to drop out every NaN that nore than 1
# z = www.drop((www[www >= 1]).index)
# print(z)

#	SECOND how to drop out every NaN that nore than 1
# missing_data = pd.concat([www], axis=1, keys=['Total'])
# www = www.drop((missing_data[missing_data['Total'] >= 1]).index)
# print(www)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(10))
# #########################################################################################
# #########################################################################################



# #dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)		# Columns with NaN's has been dropped out 
# print((missing_data[missing_data['Total'] > 1]))								# Shows columns with NaN's more then 1
# print((missing_data[missing_data['Total'] > 1]).index,1)						# Same
# print(df_train)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)	# Rows with NaN's has been dropped out
# print(df_train.head(10))
# print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...


		# FEBRUARY 1
######### 1 #########
#histogram and normal probability plot
# sns.distplot(df_train['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
#	Все, что больше 600 000 долларов можно исключить из выборки
# 	But everything's not lost

#applying log transformation
#								ЭТО АНАЛОГ НОРМАЛИЗАЦИИ
df_train['SalePrice'] = np.log(df_train['SalePrice'])
# print(df_train['SalePrice'])


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()


######### 2 #########
#histogram and normal probability plot
# sns.distplot(df_train['GrLivArea'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)

#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
# plt.show()


######### 3 #########

# A big problem because the value zero doesn't allow us to do log transformations.

#histogram and normal probability plot
# sns.distplot(df_train['TotalBsmtSF'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

# That's what I call 'high risk engineering'.
# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
# print(df_train[['HasBsmt', 'TotalBsmtSF']])

#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


######### 4 #########
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
plt.show()


######### 5 #########
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
# print(df_train.head(10))

















