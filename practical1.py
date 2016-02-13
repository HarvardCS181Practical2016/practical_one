import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
#from neural_networks import NeuralNetworks
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train.head()
df_test.head()

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()

print 'Adding features..'

"""
Feature Engineering
"""
# calculate the number of atoms for each molecule defined by smile string and adds a feature column atoms
df_all['atoms'] = df_all.apply(lambda x: Chem.MolFromSmiles(x['smiles']).GetNumAtoms(), axis=1)

# calculates the number of bonds for each molecule defined by smile string and adds a feature column bonds
df_all['bonds'] = df_all.apply(lambda x: Chem.MolFromSmiles(x['smiles']).GetNumBonds(), axis=1)

# calculates the molecular wt for each molecule defined by smile string and adds a feature column molwt
df_all['molwt'] = df_all.apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x['smiles'])), axis=1)

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape

# LR = LinearRegression()
# LR.fit(X_train, Y_train)
# LR_pred = LR.predict(X_test)

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

# SV = LinearSVR()
# SV.fit(X_train, Y_train)
# SV_pred = SV.predict(X_test)
#
# SGD = SGDRegressor()
# SGD.fit(X_train, Y_train)
# SGD_pred = SGD.predict(X_test)
#
# GBR = GradientBoostingRegressor()
# GBR.fit(X_train, Y_train)
# GBR_pred = GBR.predict(X_test)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

# write_to_file("sample1.csv", LR_pred)
write_to_file("sample2.csv", RF_pred)
# write_to_file("sample_sgd.csv", SV_pred)
# write_to_file("predicted.csv", SGD_pred)
# write_to_file("predicted.csv", GBR_pred)

df_actual = pd.read_csv("actual.csv")
print sqrt(mean_squared_error(y_true=df_actual['Prediction'], y_pred=RF_pred))
