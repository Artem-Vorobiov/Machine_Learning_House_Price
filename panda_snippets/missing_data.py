# 						      OVERVIEW

# This tutorial is divided into 6 parts:

# 1.	Pima Indians Diabetes Dataset: where we look at a dataset that has known missing values.
# 2.	Mark Missing Values: where we learn how to mark missing values in a dataset.
# 3.	Missing Values Causes Problems: where we see how a machine learning algorithm can fail when it contains missing values.
# 4.	Remove Rows With Missing Values: where we see how to remove rows that contain missing values.
# 5.	Impute Missing Values: where we replace missing values with sensible values.
# 6.	Algorithms that Support Missing Values: where we learn about algorithms that support missing values.

# The variable names are as follows:

# 0. Number of times pregnant.
# 1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
# 2. Diastolic blood pressure (mm Hg).
# 3. Triceps skinfold thickness (mm).
# 4. 2-Hour serum insulin (mu U/ml).
# 5. Body mass index (weight in kg/(height in m)^2).
# 6. Diabetes pedigree function.
# 7. Age (years).
# 8. Class variable (0 or 1).


####################                                       ####################
####################          Mark Missing Values          ####################
####################                                       ####################

import numpy
from pandas import read_csv
dataset = read_csv('pima-indians-diabetes.data.csv', header=None)
# print(dataset.describe())

# We can see that there are columns that have a minimum value of zero (0). 
# On some columns, a value of zero does not make sense and indicates an invalid or missing value.

# print((dataset[[1,2,3,4,5]] == 0))		# Everything in True and False
# print('\n\n\t\t\t NEXT \n\n\n')
# print((dataset[[1,2,3,4,5]] == 0).sum())

# We can see that columns 1,2 and 5 have just a few zero values, whereas columns 3 and 4 show a lot more, nearly half of the rows.

# This highlights that different “missing value” strategies may be needed for different columns,
# e.g. to ensure that there are still a sufficient number of records left to train a predictive model


# We can mark values as NaN easily with the Pandas DataFrame by using the replace() function on a subset of the columns we are interested in.
# After we have marked the missing values, we can use the isnull() function to mark all of the NaN values in the dataset as True 
# and get a count of the missing values for each column.

# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
print('\n\n\t\t\t NEXT \n\n\n')

# count the number of NaN values in each column
print(dataset.isnull().sum())


####################                                       ####################
####################    Missing Values Causes Problems     ####################
####################                                       ####################


#	Some algorithms don't work when there are missing values in the dataset!




#									FIRST
####################                                       ####################
####################   Remove Rows With Missing Values     ####################
####################                                       ####################

# The simplest strategy for handling missing data is to remove records that contain a missing value.

# Pandas provides the dropna() function that can be used to drop either columns or rows with missing data.

# Running this example, we can see that the number of rows has been aggressively cut from 768 in the original dataset to 392 with all rows containing a NaN removed.



#									SECOND
####################                                       ####################
####################         Impute Missing Values         ####################
####################                                       ####################

# 		There are many options we could consider when replacing a missing value, for example:

# 	1. constant value that has meaning within the domain, such as 0, distinct from all other values.
# 	2. value from another randomly selected record.
# 	3. mean, median or mode value for the column.
# 	4. value estimated by another predictive model.

# Pandas provides the fillna() function for replacing missing values with a specific value.
# 		1. For example, we can use fillna() to replace missing values with the mean value for each column.
# 		2. The scikit-learn library provides the Imputer() pre-processing class that can be used to replace missing values.
# 		3. 


#									THIRD
####################                                       ####################
############        Algorithms that Support Missing Values         ############
####################                                       ####################


#  1. There are algorithms that can be made robust to missing data, 
#  	  such as [  k-Nearest Neighbors  ] that can ignore a column from a distance measure when a value is missing.

# 2. There are also algorithms that can use the missing value as a unique and different value when building the predictive model, 
#    such as [  classification ] and [  regression trees  ].






