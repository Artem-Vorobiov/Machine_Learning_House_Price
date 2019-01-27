import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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



def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)

    df['MSZoning'] 		= df['MSZoning'].apply(zone2num_zones)
    df['Street']   		= df[ 'Street'].apply(street2num_street)
    df['Alley']    		= df['Alley'].apply(alley2num_alley)
    df['LotShape']    	= df['LotShape'].apply(LotShape2num_LotShape)
    df['LandContour']	= df['LandContour'].apply(LandContour2num_LandContour)
    df['Utilities']		= df['Utilities'].apply(Utilities2num_Utilities)
    df['LotConfig']		= df['LotConfig'].apply(LotConfig2num_LotConfig)
    df['LandSlope']		= df['LandSlope'].apply(LandSlope2num_LandSlope)

    df['Neighborhood']		= df['Neighborhood'].apply(Neighborhood2num_Neighborhood)
    # df['LotConfig']		= df['LotConfig'].apply(LotConfig2num_LotConfig)

    print(df['Neighborhood'])
    df.fillna(df.mean(), inplace=True)
    print('\n\t\t\t NEXT TURN \n\n')
    print(df['Neighborhood'])




######################.    research works.    ######################.
    # print('\n\n')
    # print(df.head(2))
    print('\n\n')
    print(df['Neighborhood'].describe())
    print('\n\n')
    print('\t\t\t Time For Error')
    print('\n\n')
##################################################################




def wrap_preprocess():
    X_train, y_train = preprocess_data("data/train.csv")

    train_size = int(len(y_train) * 0.80)

    # with h5py.File("dataset-v1.h5", 'w') as f:
    #     f.create_dataset("X_train", data=np.array(X_train[:train_size]))
    #     f.create_dataset('y_train', data=np.array(y_train[:train_size]))
    #     f.create_dataset("X_val", data=np.array(X_train[train_size:]))
    #     f.create_dataset("y_val", data=np.array(y_train[train_size:]))


wrap_preprocess()