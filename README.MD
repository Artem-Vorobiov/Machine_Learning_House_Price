# [Kaggle](https://www.kaggle.com/).

## Goal
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## File descriptions

##### train.csv 				--> _the training set_
##### test.csv  				--> _the test set_

##### data_description.txt 	--> _full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here_

##### sample_submission.csv 	--> _a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms_

##### train_model_test1.py 	--> _That is my first try, The file full of mess. Here I'm poking around with everything such as: loading SCV, transforming Categorical to Numerical, dealing with Missing Data, Normalization, choose features and labels, setting up SCV, Fitting SVM and save model with pickle_ 

##### testing1.py 			--> _The same steps as into ```train_model_test1.py```, but instead saving model we are loading fitted model, then predict data and save data into CSV_

##### compare_result.py 		--> _loading submission.csv and output.csv, merge this data into one Pandas.DataFrame within 2 columns SalePrice_x(predictied) and SalePrice_y(reality), iterate over data, compare, get absolute deviation for each row, summerize them and find out mean deviation. Which is in my case is 0.794238 for first try._


### My submission

First Attempt = 0.794238
