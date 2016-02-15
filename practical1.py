import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

from rdkit import Chem
from rdkit.Chem import Descriptors

"""
    Read in train and test as Pandas DataFrames
    """
df_train = pd.read_csv("train_features2.csv")
df_test = pd.read_csv("test_features2.csv")

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
# # calculate the number of atoms for each molecule defined by smile string and adds a feature column atoms
# df_all['atoms'] = df_all.apply(lambda x: Chem.MolFromSmiles(x['smiles']).GetNumAtoms(), axis=1)
#
# # calculates the number of bonds for each molecule defined by smile string and adds a feature column bonds
# df_all['bonds'] = df_all.apply(lambda x: Chem.MolFromSmiles(x['smiles']).GetNumBonds(), axis=1)
#
# # calculates the molecular wt for each molecule defined by smile string and adds a feature column molwt
# df_all['molwt'] = df_all.apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x['smiles'])), axis=1)

# # calculates number of valence electroms for each molecule defined by smile string and adds a feature column molwt
# df_all['valence_electron'] = df_all.apply(lambda x: Descriptors.NumValenceElectrons(Chem.MolFromSmiles(x['smiles'])), axis=1)
#
# # calculates number of double bonds for each molecule defined by smile string and adds a feature column molwt
# df_all['double_bonds'] = df_all.apply(lambda x: x['smiles'].count('=')), axis=1)


#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape

# # LinearRegression
# LR = LinearRegression()
# LR.fit(X_train, Y_train)
# LR_pred = LR.predict(X_test)
#
# # RidgeRegression
# RR = RidgeCV(cv=5)
# RR.fit(X_train, Y_train)
# RR_pred = RR.predict(X_test)
#
# # Lasso cross validation
# LSR = LassoCV(cv=5)
# LSR.fit(X_train, Y_train)
# LSR_pred = LSR.predict(X_test)

# RandomForestRegressor
RF = RandomForestRegressor(n_estimators=100, n_jobs=-1)
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

# # Linear Support Vector
# SV = LinearSVR()
# SV.fit(X_train, Y_train)
# SV_pred = SV.predict(X_test)
#
# # SGDRegressor
# SGD = SGDRegressor()
# SGD.fit(X_train, Y_train)
# SGD_pred = SGD.predict(X_test)

# GBR = GradientBoostingRegressor()
# GBR.fit(X_train, Y_train)
# GBR_pred = GBR.predict(X_test)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

# write_to_file("LR_pred2.csv", LR_pred)
# write_to_file("RR_pred2.csv", RR_pred)
# write_to_file("LSR_pred2.csv", LSR_pred)
write_to_file("RF_pred2", RF_pred)
# write_to_file("SV_pred2.csv", SV_pred)
# write_to_file("SGD_pred2.csv", SGD_pred)
# write_to_file("GBR_pred.csv", GBR_pred)

# df_actual = pd.read_csv("actual1.csv")
# print sqrt(mean_squared_error(y_true=df_actual['Prediction'], y_pred=RF_pred))
