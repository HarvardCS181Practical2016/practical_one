import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.metrics import mean_squared_error
from math import sqrt


def print_molecule_info(smiles):
    m = Chem.MolFromSmiles(smiles)
    print m.GetNumAtoms()
    print m.GetNumBonds()
    print m.GetNumHeavyAtoms()
    print Descriptors.MolWt(m)


def add_features_to_test_file(input, output):
    with open(input, 'r') as f:
        lines = f.read().splitlines()

    with open(output, 'w') as f:
        f.writelines(lines[0] + ',atom,bonds,molwt,double_bonds,valence_electrons\n')
        for line in range(1, len(lines)):
            smiles = lines[line].split(',')[1]
            m = Chem.MolFromSmiles(smiles)
            l = lines[line] + \
                ',' + str(m.GetNumAtoms()) + \
                ',' + str(m.GetNumBonds()) + \
                ',' + str(Descriptors. MolWt(m)) + \
                ',' + str(smiles.count('=')) + \
                ',' + str(Descriptors.NumValenceElectrons(m)) + '\n'
            if line % 10000 == 0:
                print line
            f.write(l)

def add_features_to_train_file(input, output):
    with open(input, 'r') as f:
        lines = f.read().splitlines()

    with open(output, 'w') as f:
        f.writelines(lines[0] + ',atom,bonds,molwt,double_bonds,valence_electrons\n')
        for line in range(1, len(lines)):
            smiles = lines[line].split(',')[0]
            m = Chem.MolFromSmiles(smiles)
            l = lines[line] + \
                ',' + str(m.GetNumAtoms()) + \
                ',' + str(m.GetNumBonds()) + \
                ',' + str(Descriptors. MolWt(m)) + \
                ',' + str(smiles.count('=')) + \
                ',' + str(Descriptors. NumValenceElectrons(m)) + '\n'
            if line % 10000 == 0:
                print line
            f.write(l)


def split_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()

    with open('train_30.csv', 'w') as f:
        f.write(lines[0])
        for line in range(700001, len(lines) - 1):
            f.write(lines[line])

    df_test = pd.read_csv("train_30.csv")
    df_test2 = df_test.drop(['gap'], axis=1)
    df_test2.to_csv('test_30.csv', index=True, index_label='id')
    df_pred = df_test['gap']
    df_pred.to_csv('actual.csv', index=True, index_label='id')


def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def print_rmse(actual_file, predicted_files):
    df_actual = pd.read_csv(actual_file)
    for pred_file in predicted_files:
        df_pred = pd.read_csv(pred_file)
        print calculate_rmse(y_true=df_actual['Prediction'], y_pred=df_pred['Prediction'])


def plot_bar_chart(file_name):
    df = pd.read_csv(file_name)
    df = df.set_index('Method', drop=True)
    print df.index
    # add some text for labels, title and axes ticks
    ax = df.plot(kind='bar', title='Regression Method vs RMSE')
    ax.set_xlabel('Regression Method')
    ax.set_ylabel('RMSE')
    ax.set_xticklabels(df.index, rotation=10)

    plt.show()


if __name__ == '__main__':
    plot_bar_chart('data.csv')
    # print_rmse('actual1.csv', ['finalSubmissio1.csv'])

# # An attemp at Neural Networks
# # sigmoid function
# def nonlin(x,deriv=False):
#     if(deriv==True):
#         return x*(1-x)
#     return 1/(1+np.exp(-x))
#
# # input dataset
# X = np.array([  [0,0,1],
#                 [0,1,1],
#                 [1,0,1],
#                 [1,1,1] ])
#
# # output dataset
# y = np.array([[0,0,1,1]]).T
#
# # seed random numbers to make calculation
# # deterministic (just a good practice)
# np.random.seed(1)
#
# # initialize weights randomly with mean 0
# syn0 = 2*np.random.random((3,1)) - 1
#
# for iter in xrange(10000):
#
#     # forward propagation
#     l0 = X
#     l1 = nonlin(np.dot(l0,syn0))
#
#     # how much did we miss?
#     l1_error = y - l1
#
#     # multiply how much we missed by the
#     # slope of the sigmoid at the values in l1
#     l1_delta = l1_error * nonlin(l1,True)
#
#     # update weights
#     syn0 += np.dot(l0.T,l1_delta)
#
# print "Output After Training:"
# print l1
