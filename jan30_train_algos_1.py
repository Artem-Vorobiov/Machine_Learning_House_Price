import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# adds
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.cluster import KMeans, MeanShift
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import pickle


def zone2num_zones(val):
	if 'A' == val:
		return 0
	elif 'C' == val:
		return 1
	elif 'FV' == val:
		return 2
	elif 'I' == val:
		return 3
	elif 'RH' == val:
		return 4
	elif 'RL' == val:
		return 5
	elif 'RP' == val:
		return 6
	elif 'RM' == val:
		return 7

def street2num_street(val):
	if 'Pave'== val:
		return 0
	else:
		return 1

def alley2num_alley(val):
	if 'Grvl' == val:
		return 2
	elif 'Pave' == val:
		return 1
	else:
		return 0

def LotShape2num_LotShape(val):
	if 'Reg' == val:
		return 0
	elif 'IR1' == val:
		return 1
	elif 'IR2' == val:
		return 2
	elif 'IR3' == val:
		return 3


def LandContour2num_LandContour(val):
	if 'Lvl' == val:
		return 0
	elif 'Bnk' == val:
		return 1
	elif 'HLS' == val:
		return 2
	elif 'Low' == val:
		return 3

def Utilities2num_Utilities(val):
	if 'AllPub' == val:
		return 0
	elif 'NoSewr' == val:
		return 1
	elif 'NoSeWa' == val:
		return 2
	elif 'ELO' == val:
		return 3

def LotConfig2num_LotConfig(val):
	if 'Inside' == val:
		return 0
	elif 'Corner' == val:
		return 1
	elif 'CulDSac' == val:
		return 2
	elif 'FR2' == val:
		return 3
	elif 'FR3' == val:
		return 4

def LandSlope2num_LandSlope(val):
	if 'Gtl' == val:
		return 0
	elif 'Mod' == val:
		return 1
	elif 'Sev' == val:
		return 2

def Neighborhood2num_Neighborhood(val):
	if 'Blmngtn' == val:
		return 0
	elif 'Blueste' == val:
		return 1
	elif 'BrDale' == val:
		return 2
	elif 'BrkSide' == val:
		return 3
	elif 'ClearCr' == val:
		return 4
	elif 'CollgCr' == val:
		return 5
	elif 'Crawfor' == val:
		return 6
	elif 'Edwards' == val:
		return 8
	elif 'Gilbert' == val:
		return 9
	elif 'IDOTRR' == val:
		return 10
	elif 'MeadowV' == val:
		return 11
	elif 'Mitchel' == val:
		return 12
	elif 'NoRidge' == val:
		return 13
	elif 'NPkVill' == val:
		return 14
	elif 'NridgHt' == val:
		return 15
	elif 'NWAmes' == val:
		return 16
	elif 'OldTown' == val:
		return 17
	elif 'SWISU' == val:
		return 18
	elif 'Sawyer' == val:
		return 19
	elif 'SawyerW' == val:
		return 20
	elif 'Somerst' == val:
		return 21
	elif 'StoneBr' == val:
		return 22
	elif 'Timber' == val:
		return 23
	elif 'Veenker' == val:
		return 24

def Condition1_2num_Condition1(val):
	if 'Artery' == val:
		return 0
	elif 'Feedr' == val:
		return 1
	elif 'Norm' == val:
		return 2
	elif 'RRNn' == val:
		return 3
	elif 'RRAn' == val:
		return 4
	elif 'PosN' == val:
		return 5
	elif 'PosA' == val:
		return 6
	elif 'RRNe' == val:
		return 8
	elif 'RRAe' == val:
		return 9

def Condition2_2num_Condition2(val):
	if 'Artery' == val:
		return 0
	elif 'Feedr' == val:
		return 1
	elif 'Norm' == val:
		return 2
	elif 'RRNn' == val:
		return 3
	elif 'RRAn' == val:
		return 4
	elif 'PosN' == val:
		return 5
	elif 'PosA' == val:
		return 6
	elif 'RRNe' == val:
		return 8
	elif 'RRAe' == val:
		return 9

def BldgType2num_BldgType(val):
	if '1Fam' == val:
		return 0
	elif '2FmCon' == val:
		return 1
	elif 'Duplx' == val:
		return 2
	elif 'TwnhsE' == val:
		return 3
	elif 'TwnhsI' == val:
		return 4

def HouseStyle2num_HouseStyle(val):
	if '1Story' == val:
		return 0
	elif '1.5Fin' == val:
		return 1
	elif '1.5Unf' == val:
		return 2
	elif '2Story' == val:
		return 3
	elif '2.5Fin' == val:
		return 4
	elif '2.5Unf' == val:
		return 5
	elif 'SFoyer' == val:
		return 6
	elif 'SLvl' == val:
		return 8

def RoofStyle2num_RoofStyle(val):
	if 'Flat' == val:
		return 0
	elif 'Gable' == val:
		return 1
	elif 'Gambrel' == val:
		return 2
	elif 'Hip' == val:
		return 3
	elif 'Mansard' == val:
		return 4
	elif 'Shed' == val:
		return 5

def RoofMatl2num_RoofMatl(val):
	if 'ClyTile' == val:
		return 0
	elif 'CompShg' == val:
		return 1
	elif 'Membran' == val:
		return 2
	elif 'Metal' == val:
		return 3
	elif 'Roll' == val:
		return 4
	elif 'Tar&Grv' == val:
		return 5
	elif 'WdShake' == val:
		return 6
	elif 'WdShngl' == val:
		return 8


def Exterior1st2num_Exterior1st(val):
	if 'AsbShng' == val:
		return 0
	elif 'AsphShn' == val:
		return 1
	elif 'BrkComm' == val:
		return 2
	elif 'BrkFace' == val:
		return 3
	elif 'CBlock' == val:
		return 4
	elif 'CemntBd' == val:
		return 5
	elif 'HdBoard' == val:
		return 6
	elif 'ImStucc' == val:
		return 8
	elif 'MetalSd' == val:
		return 9
	elif 'Other' == val:
		return 10
	elif 'Plywood' == val:
		return 11
	elif 'PreCast' == val:
		return 12
	elif 'Stone' == val:
		return 13
	elif 'Stucco' == val:
		return 14
	elif 'VinylSd' == val:
		return 15
	elif 'Wd Sdng' == val:
		return 16
	elif 'WdShing' == val:
		return 17

def Exterior2nd2num_Exterior2nd(val):
	if 'AsbShng' == val:
		return 0
	elif 'AsphShn' == val:
		return 1
	elif 'BrkComm' == val:
		return 2
	elif 'BrkFace' == val:
		return 3
	elif 'CBlock' == val:
		return 4
	elif 'CemntBd' == val:
		return 5
	elif 'HdBoard' == val:
		return 6
	elif 'ImStucc' == val:
		return 8
	elif 'MetalSd' == val:
		return 9
	elif 'Other' == val:
		return 10
	elif 'Plywood' == val:
		return 11
	elif 'PreCast' == val:
		return 12
	elif 'Stone' == val:
		return 13
	elif 'Stucco' == val:
		return 14
	elif 'VinylSd' == val:
		return 15
	elif 'Wd Sdng' == val:
		return 16
	elif 'WdShing' == val:
		return 17

def MasVnrType2num_MasVnrType(val):
	if 'BrkCmn' == val:
		return 0
	elif 'BrkFace' == val:
		return 1
	elif 'CBlock' == val:
		return 2
	elif 'None' == val:
		return 3
	elif 'Stone' == val:
		return 4

def ExterQual2num_ExterQual(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def ExterCond2num_ExterCond(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def Foundation2num_Foundation(val):
	if 'BrkTil' == val:
		return 0
	elif 'CBlock' == val:
		return 1
	elif 'PConc' == val:
		return 2
	elif 'Slab' == val:
		return 3
	elif 'Stone' == val:
		return 4
	elif 'Wood' == val:
		return 5

def BsmtQual2num_BsmtQual(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def BsmtCond2num_BsmtCond(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def BsmtExposure2num_BsmtExposure(val):
	if 'Gd' == val:
		return 0
	elif 'Av' == val:
		return 1
	elif 'Mn' == val:
		return 2
	elif 'No' == val:
		return 3
	elif 'NA' == val:
		return 4

def BsmtFinType12num_BsmtFinType1(val):
	if 'GLQ' == val:
		return 0
	elif 'ALQ' == val:
		return 1
	elif 'BLQ' == val:
		return 2
	elif 'Rec' == val:
		return 3
	elif 'LwQ' == val:
		return 4
	elif 'Unf' == val:
		return 5
	elif 'NA' == val:
		return 6

def BsmtFinType22num_BsmtFinType2(val):
	if 'GLQ' == val:
		return 0
	elif 'ALQ' == val:
		return 1
	elif 'BLQ' == val:
		return 2
	elif 'Rec' == val:
		return 3
	elif 'LwQ' == val:
		return 4
	elif 'Unf' == val:
		return 5
	elif 'NA' == val:
		return 6

def Heating2num_Heating(val):
	if 'Floor' == val:
		return 0
	elif 'GasA' == val:
		return 1
	elif 'GasW' == val:
		return 2
	elif 'Grav' == val:
		return 3
	elif 'OthW' == val:
		return 4
	elif 'Wall' == val:
		return 5

def HeatingQC2num_HeatingQC(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def CentralAir2num_CentralAir(val):
	if 'N' == val:
		return 0
	elif 'Y' == val:
		return 1

def Electrical2num_Electrical(val):
	if 'SBrkr' == val:
		return 0
	elif 'FuseA' == val:
		return 1
	elif 'FuseF' == val:
		return 2
	elif 'FuseP' == val:
		return 3
	elif 'Mix' == val:
		return 4

def KitchenQual2num_KitchenQual(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4

def Functional2num_Functional(val):
	if 'Typ' == val:
		return 0
	elif 'Min1' == val:
		return 1
	elif 'Min2' == val:
		return 2
	elif 'Mod' == val:
		return 3
	elif 'Maj1' == val:
		return 4
	elif 'Maj2' == val:
		return 5
	elif 'Sev' == val:
		return 6
	elif 'Sal' == val:
		return 7

def FireplaceQu2num_FireplaceQu(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4
	elif 'NA' == val:
		return 5

def GarageType2num_GarageType(val):
	if '2Types' == val:
		return 0
	elif 'Attchd' == val:
		return 1
	elif 'Basment' == val:
		return 2
	elif 'BuiltIn' == val:
		return 3
	elif 'CarPort' == val:
		return 4
	elif 'Detchd' == val:
		return 5
	elif 'NA' == val:
		return 6

def GarageFinish2num_GarageFinish(val):
	if 'Fin' == val:
		return 0
	elif 'RFn' == val:
		return 1
	elif 'Unf' == val:
		return 2
	elif 'NA' == val:
		return 3

def GarageQual2num_GarageQual(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4
	elif 'NA' == val:
		return 5

def GarageCond2num_GarageCond(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'Po' == val:
		return 4
	elif 'NA' == val:
		return 5

def PavedDrive2num_PavedDrive(val):
	if 'Y' == val:
		return 0
	elif 'P' == val:
		return 1
	elif 'N' == val:
		return 2

def PoolQC2num_PoolQC(val):
	if 'Ex' == val:
		return 0
	elif 'Gd' == val:
		return 1
	elif 'TA' == val:
		return 2
	elif 'Fa' == val:
		return 3
	elif 'NA' == val:
		return 4

def Fence2num_Fence(val):
	if 'GdPrv' == val:
		return 0
	elif 'MnPrv' == val:
		return 1
	elif 'GdWo' == val:
		return 2
	elif 'MnWw' == val:
		return 3
	elif 'NA' == val:
		return 4

def MiscFeature2num_MiscFeature(val):
	if 'Elev' == val:
		return 0
	elif 'Gar2' == val:
		return 1
	elif 'Othr' == val:
		return 2
	elif 'Shed' == val:
		return 3
	elif 'TenC' == val:
		return 4
	elif 'NA' == val:
		return 5

def SaleType2num_SaleType(val):
	if 'WD' == val:
		return 0
	elif 'CWD' == val:
		return 1
	elif 'VWD' == val:
		return 2
	elif 'New' == val:
		return 3
	elif 'COD' == val:
		return 4
	elif 'Con' == val:
		return 5
	elif 'ConLw' == val:
		return 6
	elif 'ConLI' == val:
		return 7
	elif 'ConLD' == val:
		return 8
	elif 'Oth' == val:
		return 9

def SaleCondition2num_SaleCondition(val):
	if 'Normal' == val:
		return 0
	elif 'Abnorml' == val:
		return 1
	elif 'AdjLand' == val:
		return 2
	elif 'Alloca' == val:
		return 3
	elif 'Family' == val:
		return 4
	elif 'Partial' == val:
		return 5



def preprocess_data(csv_file):

    df = pd.read_csv(csv_file)

######################.    research works.    ######################.
    # print(df.isnull().sum())
    # print(df.isnull())
    # count = 0
    # for f in df.isnull().sum():
    # 	if f != 0:
	   #  	print(f)
	   #  	count += 1
    # print('\n\t\t Cool. It is  [ {} ] columns with NaN'.format(count))
    # print(count)
##################################################################

	# Categorical to Numerical
    df['MSZoning'] 			= df['MSZoning'].apply(zone2num_zones)
    df['Street']   			= df[ 'Street'].apply(street2num_street)
    df['Alley']    			= df['Alley'].apply(alley2num_alley)
    df['LotShape']	    	= df['LotShape'].apply(LotShape2num_LotShape)
    df['LandContour']		= df['LandContour'].apply(LandContour2num_LandContour)
    df['Utilities']			= df['Utilities'].apply(Utilities2num_Utilities)
    df['LotConfig']			= df['LotConfig'].apply(LotConfig2num_LotConfig)
    df['LandSlope']			= df['LandSlope'].apply(LandSlope2num_LandSlope)
    df['Neighborhood']		= df['Neighborhood'].apply(Neighborhood2num_Neighborhood)
    df['Condition1']		= df['Condition1'].apply(Condition1_2num_Condition1)
    df['Condition2']		= df['Condition2'].apply(Condition2_2num_Condition2)
    df['BldgType']		    = df['BldgType'].apply(BldgType2num_BldgType)
    df['HouseStyle']		= df['HouseStyle'].apply(HouseStyle2num_HouseStyle)
    df['RoofStyle']			= df['RoofStyle'].apply(RoofStyle2num_RoofStyle)
    df['RoofMatl']			= df['RoofMatl'].apply(RoofMatl2num_RoofMatl)
    df['Exterior1st']		= df['Exterior1st'].apply(Exterior1st2num_Exterior1st)
    df['Exterior2nd']		= df['Exterior2nd'].apply(Exterior2nd2num_Exterior2nd)
    df['MasVnrType']		= df['MasVnrType'].apply(MasVnrType2num_MasVnrType)
    df['ExterQual']			= df['ExterQual'].apply(ExterQual2num_ExterQual)
    df['ExterCond']			= df['ExterCond'].apply(ExterCond2num_ExterCond)
    df['Foundation']		= df['Foundation'].apply(Foundation2num_Foundation)
    df['BsmtQual']			= df['BsmtQual'].apply(BsmtQual2num_BsmtQual)
    df['BsmtCond']			= df['BsmtCond'].apply(BsmtCond2num_BsmtCond)
    df['BsmtExposure']		= df['BsmtExposure'].apply(BsmtExposure2num_BsmtExposure)
    df['BsmtFinType1']		= df['BsmtFinType1'].apply(BsmtFinType12num_BsmtFinType1)
    df['BsmtFinType2']		= df['BsmtFinType2'].apply(BsmtFinType22num_BsmtFinType2)
    df['Heating']			= df['Heating'].apply(Heating2num_Heating)
    df['HeatingQC']			= df['HeatingQC'].apply(HeatingQC2num_HeatingQC)
    df['CentralAir']		= df['CentralAir'].apply(CentralAir2num_CentralAir)
    df['Electrical']		= df['Electrical'].apply(Electrical2num_Electrical)
    df['KitchenQual']		= df['KitchenQual'].apply(KitchenQual2num_KitchenQual)
    df['Functional']		= df['Functional'].apply(Functional2num_Functional)
    df['FireplaceQu']		= df['FireplaceQu'].apply(FireplaceQu2num_FireplaceQu)
    df['GarageType']		= df['GarageType'].apply(GarageType2num_GarageType)

    # df['']		= df[''].apply( 2num_)
    df['GarageFinish']		= df['GarageFinish'].apply(GarageFinish2num_GarageFinish)
    df['GarageQual']		= df['GarageQual'].apply(GarageQual2num_GarageQual)
    df['GarageCond']		= df['GarageCond'].apply(GarageCond2num_GarageCond)
    df['PavedDrive']		= df['PavedDrive'].apply(PavedDrive2num_PavedDrive)

    df['PoolQC']			= df['PoolQC'].apply(PoolQC2num_PoolQC)
    df['Fence']				= df['Fence'].apply(Fence2num_Fence)
    df['MiscFeature']		= df['MiscFeature'].apply(MiscFeature2num_MiscFeature)
    df['SaleType']			= df['SaleType'].apply(SaleType2num_SaleType)
    df['SaleCondition']		= df['SaleCondition'].apply(SaleCondition2num_SaleCondition)
   
    # Missing Data
    df.fillna(df.mean(), inplace=True)

    # Normalization 43
    scaler = MinMaxScaler()

    # df['MSZoning'] = scaler.fit_transform(np.array(df['MSZoning']).reshape(-1, 1)) 		# <class 'pandas.core.series.Series'>
    # df['Street'] = scaler.fit_transform(np.array(df['Street']).reshape(-1, 1))
    # df['Alley'] = scaler.fit_transform(np.array(df['Alley']).reshape(-1, 1))
    # df['LotShape'] = scaler.fit_transform(np.array(df['LotShape']).reshape(-1, 1))
    # df['LandContour'] = scaler.fit_transform(np.array(df['LandContour']).reshape(-1, 1))
    # df['Utilities'] = scaler.fit_transform(np.array(df['Utilities']).reshape(-1, 1))
    # df['LotConfig'] = scaler.fit_transform(np.array(df['LotConfig']).reshape(-1, 1))
    # df['LandSlope'] = scaler.fit_transform(np.array(df['LandSlope']).reshape(-1, 1))
    # df['Neighborhood'] = scaler.fit_transform(np.array(df['Neighborhood']).reshape(-1, 1))
    # df['Condition1'] = scaler.fit_transform(np.array(df['Condition1']).reshape(-1, 1))
    # df['Condition2'] = scaler.fit_transform(np.array(df['Condition2']).reshape(-1, 1))
    # df['BldgType'] = scaler.fit_transform(np.array(df['BldgType']).reshape(-1, 1))
    # df['HouseStyle'] = scaler.fit_transform(np.array(df['HouseStyle']).reshape(-1, 1))

    # df['RoofStyle'] = scaler.fit_transform(np.array(df['RoofStyle']).reshape(-1, 1))
    # df['RoofMatl'] = scaler.fit_transform(np.array(df['RoofMatl']).reshape(-1, 1))
    # df['Exterior1st'] = scaler.fit_transform(np.array(df['Exterior1st']).reshape(-1, 1))
    # df['Exterior2nd'] = scaler.fit_transform(np.array(df['Exterior2nd']).reshape(-1, 1))
    # df['MasVnrType'] = scaler.fit_transform(np.array(df['MasVnrType']).reshape(-1, 1))
    # df['ExterQual'] = scaler.fit_transform(np.array(df['ExterQual']).reshape(-1, 1))
    # df['ExterCond'] = scaler.fit_transform(np.array(df['ExterCond']).reshape(-1, 1))
    # df['Foundation'] = scaler.fit_transform(np.array(df['Foundation']).reshape(-1, 1))
    # df['BsmtQual'] = scaler.fit_transform(np.array(df['BsmtQual']).reshape(-1, 1))
    # df['BsmtCond'] = scaler.fit_transform(np.array(df['BsmtCond']).reshape(-1, 1))

    # df['BsmtExposure'] = scaler.fit_transform(np.array(df['BsmtExposure']).reshape(-1, 1))
    # df['BsmtFinType1'] = scaler.fit_transform(np.array(df['BsmtFinType1']).reshape(-1, 1))
    # df['BsmtFinType2'] = scaler.fit_transform(np.array(df['BsmtFinType2']).reshape(-1, 1))
    # df['Heating'] = scaler.fit_transform(np.array(df['Heating']).reshape(-1, 1))
    # df['HeatingQC'] = scaler.fit_transform(np.array(df['HeatingQC']).reshape(-1, 1))
    # df['CentralAir'] = scaler.fit_transform(np.array(df['CentralAir']).reshape(-1, 1))
    # df['Electrical'] = scaler.fit_transform(np.array(df['Electrical']).reshape(-1, 1))

    # df['KitchenQual'] = scaler.fit_transform(np.array(df['KitchenQual']).reshape(-1, 1))
    # df['Functional'] = scaler.fit_transform(np.array(df['Functional']).reshape(-1, 1))
    # df['FireplaceQu'] = scaler.fit_transform(np.array(df['FireplaceQu']).reshape(-1, 1))
    # df['GarageType'] = scaler.fit_transform(np.array(df['GarageType']).reshape(-1, 1))
    # df['GarageFinish'] = scaler.fit_transform(np.array(df['GarageFinish']).reshape(-1, 1))
    # df['GarageQual'] = scaler.fit_transform(np.array(df['GarageQual']).reshape(-1, 1))
    # df['GarageCond'] = scaler.fit_transform(np.array(df['GarageCond']).reshape(-1, 1))
    # df['PavedDrive'] = scaler.fit_transform(np.array(df['PavedDrive']).reshape(-1, 1))

    # df['PoolQC'] = scaler.fit_transform(np.array(df['PoolQC']).reshape(-1, 1))
    # df['Fence'] = scaler.fit_transform(np.array(df['Fence']).reshape(-1, 1))
    # df['MiscFeature'] = scaler.fit_transform(np.array(df['MiscFeature']).reshape(-1, 1))
    # df['SaleType'] = scaler.fit_transform(np.array(df['SaleType']).reshape(-1, 1))
    # df['SaleCondition'] = scaler.fit_transform(np.array(df['SaleCondition']).reshape(-1, 1))
    # print(type(df['MSZoning']))

######################.    Code snippet for describe() .    ######################.
######################.       Standardize Evrything.        ######################.
######################.    		     Usefull .   		    ######################.
    # print('\n\n Intermediate Result')
    count = 0
    all_max = []
    max_max = []
    target_cols = ['SalePrice']
    features_cols = []

    for f in df:
    	count += 1
    	maxx = []
    	if f != 'SalePrice':
    		features_cols.append(f)
    	# 	df['{}'.format(f)] = scaler.fit_transform(np.array(df['{}'.format(f)]).reshape(-1, 1))
	    # 	for i in df['{}'.format(f)].describe():
	    # 		if i != 1460:
	    # 			maxx.append(i)
    	# # print(max(maxx))
	    # 	all_max.append(max(maxx))
    # print('\n\tThe End\n')
    # print(count)
    # print(len(all_max))
    # print('\n\n If more then 1')
    for b in all_max:
    	if b > 1:
    		max_max.append(b)
    # print(max_max)
    # print(len(max_max))
    # print(df.describe())
    features_cols.remove('Id')
    # print(features_cols)
    y_train = df[target_cols]
    # X_train = df[['YearBuilt','LotFrontage', 'MSSubClass', 'GarageArea', 'LotArea', 'YearBuilt', 'BsmtFinSF1', \
    #  'YearRemodAdd', 'MasVnrArea',  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'YrSold', \
    #  'BsmtFinSF2', 'Electrical', 'LowQualFinSF', 'FullBath', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', \
    #  'PoolArea', 'MiscFeature', 'SaleType']]
    # # X_train = df[features_cols]
    X_train = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
    # X_train = df[features_cols]
    return X_train, y_train
##################################################################


######################.    research works.    ######################.
    # print('\n\t\t\t NEXT TURN \n\n')
    # print(df['Neighborhood'])
    # www = df.describe()
    # for f in www:
    # 	if f != 1460:
    # 		print(f)
    # 		print(type(f))

    # print(type(www))					#	<class 'pandas.core.frame.DataFrame'>
    # df.to_csv("test_description_2.csv")
    # print(df.isnull().sum())
    # print(df.describe())
    print('\n\n')
    # print(df.head(2))
    # print('\n\n')
    # print(df[['HouseStyle', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle'\
    # 	, 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', \
    # 	'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', \
    # 	'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',\
    # 	'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', \
    # 	'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', \
    # 	'SaleCondition']].describe())
    # print('\n\n')
    # print('\t\t\t Time For Error')
    # print('\n\n')
##################################################################






def wrap_preprocess():
    X, y = preprocess_data("data/train.csv")

    # X = X.reshape(-1, 1)
    # X = np.ravel(X_train)
    # y = np.ravel(y_train)
    # train_size = int(len(y_train) * 0.80)
    
    # Xtrain = np.array(X_train[:train_size])
    # ytrain = np.array(y_train[:train_size])
    # X_val = np.array(X_train[train_size:])
    # y_val = np.array(y_train[train_size:])
    # print(X)

    # www = np.ravel(Xtrain)
    # print(www.shape)
##################################################################
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    # print(X_train)
    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)
    # y_train = preprocessing.scale(y_train)
    # y_test = preprocessing.scale(y_test)
    # clf = svm.SVR()
    # clf = LinearRegression()
    # clf = LinearRegression(n_jobs=-1)

####	####	####	####	####
    # clf = svm.SVR(kernel='linear')				# Accuracy = 0.6066; = 0.74;  = 0.73; 
# Before I had acc=0.49 and 4 columns. Now I have acc=0.60 and 10 columns
	####	####	####	####	####

####	####	####	####	####
    # clf = DecisionTreeRegressor(random_state=1)	# Accuracy = 0.6861; acc = 0.75	
    ####	####	####	####	####

####	####	####	####	####
    # clf = LinearRegression()						# Accuracy = 0.6604; acc = 0.68
    ####	####	####	####	####

####	####	####	####	####
    # clf = neighbors.KNeighborsClassifier()		# Accuracy = 0.003 acc = 0.006
    ####	####	####	####	####

####	####	####	####	####
    # clf = KMeans(n_clusters=2)							#	 READ ABOUT IT
    ####	####	####	####	####

####	####	####	####	####
    # clf = MeanShift()										#	 READ ABOUT IT
    ####	####	####	####	####

####	####	####	####	####
    # clf = LogisticRegression()	# Accuracy = 0.01369; acc = 0.0068	
    ####	####	####	####	####

####	####	####	####	####
    # clf = MLPClassifier()	# Accuracy = 0.0136 acc = 0.0034
    ####	####	####	####	####

####	####	####	####	####
    # clf_1 = tree.DecisionTreeClassifier(max_depth=2)	# Accuracy = 0.0068 acc = 
    # clf_2 = tree.DecisionTreeClassifier(max_depth=5)
    ####	####	####	####	####

# ####	####	####	####	####
    # clf_1 = RandomForestRegressor(n_estimators = 1000, random_state = 42)	# Accuracy = 0.8430 acc = 
    clf_1 = RandomForestRegressor()											# Accuracy = 00.8346 acc = 
#     ####	####	####	####	####

####	####	####	####	####
    # clf_1 = GaussianNB()							# Accuracy = 0.8430 acc = 
    ####	####	####	####	####

####	####	####	####	####
    # clf_1 = AdaBoostClassifier()							# Accuracy = 0.0068 acc = 
    ####	####	####	####	####

    # for k in ['linear','poly','rbf','sigmoid']:
    # 	clf = svm.SVR(kernel=k)
    # 	clf.fit(X_train, y_train)
    # 	confidence = clf.score(X_test, y_test)
    # 	print(k,confidence)

    clf_1.fit(X_train, y_train)
    # clf_2.fit(X_train, y_train)
    confidence_1 = clf_1.score(X_test, y_test)
    # confidence_2 = clf_2.score(X_test, y_test)
    print('\n\t\t CONFIDENCE - 1\n')
    print(confidence_1)
    # print('\n\t\t CONFIDENCE - 2\n')
    # print(confidence_2)

    filename = 'models/jan31_RandomForestRegressor_HeatMap.sav'
    pickle.dump(clf_1, open(filename, 'wb'))
    # filename = 'models/jan30_10col_RandomForestRegressor_2.sav'
    # pickle.dump(clf_2, open(filename, 'wb'))


##################################################################

    
    # with h5py.File("dataset-v1.h5", 'w') as f:
    #     f.create_dataset("X_train", data=np.array(X_train[:train_size]))
    #     f.create_dataset('y_train', data=np.array(y_train[:train_size]))
    #     f.create_dataset("X_val", data=np.array(X_train[train_size:]))
    #     f.create_dataset("y_val", data=np.array(y_train[train_size:]))


wrap_preprocess()