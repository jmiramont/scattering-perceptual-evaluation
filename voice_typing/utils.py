import torch
from kymatio.torch import Scattering1D
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import decimal
from sklearn import svm
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# sys.path.append('/content/gdrive/MyDrive/scatter_signal_typing')
import torch
from kymatio.torch import Scattering1D



def dic2df(midic):
    """
    This function transforms a dictionary of arbitrary depth into a pandas' DataFrame object.
    """
    auxdic = dict()
    for key in midic:
        if isinstance(midic[key], dict):
            df = dic2df(midic[key])
            auxdic[key] = df       
        else:
            return pd.DataFrame(midic)
    
    # print(auxdic)
    df = pd.concat(auxdic,axis = 0)
    # print(df)
    return df

# Functions for scattering coefficients
log_eps = 1e-6
# Total variation features:
def tv_feats(X,g):
    if g == 0:
      X = np.log(np.abs(X)+log_eps)
    else:
      X = np.power(np.abs(X)+log_eps, g)
    
    X = np.abs(np.diff(X))
    X = np.sum(X, axis=2)
    return X


def mean_coeffs(X,g):
    log_eps = 1e-6
    # X = np.log(np.abs(X)+log_eps)
    if g == 0:
      X = np.log(np.abs(X)+log_eps)
    else:
      X = np.power(np.abs(X)+log_eps, g)
    
    media = np.mean(X, axis = 2)
    devstd = np.std(X, axis = 2)
    
    return media, devstd


def label_correlation(x1,x2):
    x2 = x2.reshape(-1,1)
    x = np.concatenate((x1,x2), axis = 1)
    R = np.corrcoef(x, y=None, rowvar = False)
    return R

def get_feature_vector(data, J, g, order_coeffs, type_feats):
    T = np.max(data.shape)
    scattering = Scattering1D(J, T, 1)
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
      scattering.cuda()
      data = data.cuda()
    
    sx1 = scattering.forward(data)
    if use_cuda:
      sx1 = sx1.cpu()

    sx  = sx1.numpy()
    media, devstd = mean_coeffs(sx,g)
    sx = tv_feats(sx, g)

    media_dic = {
        'order0': media[:,order0[0]],
        'order1': media[:,order1[0]],
        'order2': media[:,order2[0]],
        }

    std_dic = {
        'order0': devstd[:,order0[0]],
        'order1': devstd[:,order1[0]],
        'order2': devstd[:,order2[0]],
        }

    sx_dic = {
        'order0': sx[:,order0[0]],
        'order1': sx[:,order1[0]],
        'order2': sx[:,order2[0]],
        }


    features_dic = {
        'mean' : np.concatenate([media_dic[orders] for orders in order_coeffs], axis=1),
        'sd' : np.concatenate([std_dic[orders] for orders in order_coeffs], axis=1),
        'tv' : np.concatenate([sx_dic[orders] for orders in order_coeffs], axis=1),
            }    

    # media = media[:,np.concatenate((order0[0], order2[0]))]
    # devstd = devstd[:,np.concatenate((order0[0], order2[0]))]
    # sx = sx[:,np.concatenate((order0[0], order2[0]))]

    # sx = media
    # sx = np.concatenate((sx, media), axis=1)
    # sx = np.nan_to_num(sx)
    feats = np.concatenate([features_dic[types] for types in type_feats], axis=1)
    return feats


def hp_fun(X,y):
    """
    Feature extraction and clasification for hyperparameters estimation.
    Input:
      X: Data matrix, each column corresponds to one variable or feature.
      y: Label vector, each row corresponds to the label of the corresponding row in X.
      K: Number of folds for cross-validation, we use five-folds c-v for hp.
  
    Output:
      mean_acc: Mean accuracy for model selection.
    """
    results = dict()
    K = 5 # Five-folds cross-validation for hyperparameters
    acc = np.zeros((K, 1))
    scaler = StandardScaler()  # Data normalization function
    model = svm.SVC(kernel='rbf') # SVM for clasification
    # Other models
    # model = KNeighborsClassifier(n_neighbors = 27, weights='distance')
    
    skf = model_selection.StratifiedKFold(n_splits=K)
    skf.get_n_splits(X) 
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        # print(k)
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train.ravel())
        y_hat = model.predict(X_test)
        acc[k] = accuracy_score(y_test, y_hat)

    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    results['accuracy'] = (mean_acc,std_acc)
    return results

def val_fun(X,y, Xc = None,yc = None):
    """
    Feature extraction and clasification for model validation.
    Input:
      X: Data matrix, each column corresponds to one variable or feature.
      y: Label vector, each row corresponds to the label of the corresponding row in X.
      K: Number of folds for cross-validation, we use five-folds c-v for hp.
      Xc: Alternative test set for cross-dataset trials.
      yc: Alternative test set's labels.
    Output:
      mean_acc: Mean accuracy for model selection.
    """
    K = 10 #folds for CV
    skf = model_selection.StratifiedKFold(n_splits=K)  
    acc = np.zeros((K,1))
    acc_cross = np.zeros((K,1))
    cm = np.zeros((3,3))
    cm_cross = np.zeros((3,3))

    scaler = StandardScaler()  # Data normalization function
    model = svm.SVC(kernel='rbf') # SVM for clasification
    # model = KNeighborsClassifier(n_neighbors = 27, weights='distance')
    skf.get_n_splits(X)

    results = pd.DataFrame( columns = ['Acc', 'Acc_Cross.'] +
                           ['cm_{}'.format(i) for i in range(9)] +
                           ['cm_cross{}'.format(i) for i in range(9)])

    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train.ravel())
        # print(1/X_train.shape[1]*X_train.var())
        y_hat = model.predict(X_test)

        if type(Xc) is np.ndarray:
          scaler.fit(Xc)
          Xc = scaler.transform(Xc)
          yc_hat = model.predict(Xc)
          acc_cross[k] = accuracy_score(yc, yc_hat)
          cm_cross = confusion_matrix(yc, yc_hat, normalize='true')

        acc[k] = accuracy_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat, normalize='true')
        results.head()
        results.loc[k] = ([acc[k,0], acc_cross[k,0]] + 
                          np.concatenate([i for i in cm]).tolist() + 
                          np.concatenate([i for i in cm_cross]).tolist())
                                
    mean_acc = np.mean(acc)

    if type(Xc) is np.ndarray:
      mean_acc_cross = np.mean(acc_cross)
      return mean_acc, results, mean_acc_cross
    else:
      return mean_acc, results

