import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#############	COMPREHENSIVE DATA EXPLORATION WITH PYTHON
#############	https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


########## Set up display image ##########
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20


########## Load data ##########
train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')


quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
# for f in train.columns:
# 	print(train.dtypes[f])		# Shows us if it Object or Digit
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
# print(len(quantitative))


########## Missing Data ##########
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


########## Histograms ##########
# Does SalePrice follow normal distribution, 
import scipy.stats as st
y = train['SalePrice']
plt.figure(1)
plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2) 
plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3)
plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
# plt.show()
# It is apparent that SalePrice doesn't follow normal distribution, 
# so before performing regression it has to be transformed


########## Histograms ##########
########## Checking ##########
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01			# <function <lambda> at 0x1040088c8>
normal = pd.DataFrame(train[quantitative])
normal = normal.apply(test_normality)
# print(not normal.any())

print('\n\n')
# print(normal)
# print(normal.head(20))

f = pd.melt(train, value_vars=quantitative)

# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
# g = g.map(sns.distplot, "value")

































