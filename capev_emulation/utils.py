import numpy as np
import os
from scipy.io import wavfile
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import math

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor


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

def rmse_between_raters(y_raters,y_hat):
  # print(y_hat.shape)
  # print(y_raters.shape)
  rmse_out = list()
  for j in range(y_raters.shape[1]):
    rmse_out.append(np.sqrt(mean_squared_error(y_raters[:,j], y_hat)))
  # print(rmse_out)
  return rmse_out

def hp_fun(X,y,K,sev = None, rou = None, bre = None, srt = None, pit = None, lou = None):
    """
    Basic performance estimation function.
    Input:
      X: Data matrix, each column corresponds to one variable or feature.
      y: Label vector, each row corresponds to the label of the corresponding row in X.
      K: Number of folds for cross-validation.
  
    Output:
      mean_acc: Mean accuracy for model selection.
    """

    kf = model_selection.KFold(n_splits=K)
    kf.get_n_splits(X)
    scaler = StandardScaler()  # Data normalization function

    R_sev = np.zeros((K,7,7))
    R_rou = np.zeros((K,7,7))
    R_bre = np.zeros((K,7,7))
    R_srt = np.zeros((K,7,7))
    R_pit = np.zeros((K,7,7))
    R_lou = np.zeros((K,7,7))
    
    RMSE_sev = np.zeros((K,6))
    RMSE_rou = np.zeros((K,6))
    RMSE_bre = np.zeros((K,6))
    RMSE_srt = np.zeros((K,6))
    RMSE_pit = np.zeros((K,6))
    RMSE_lou = np.zeros((K,6))

    mse = np.zeros((K, 1))
    acc = np.zeros((K, y.shape[1]))
    mout = np.zeros((K, y.shape[1]))
    r2mout = np.zeros((K, y.shape[1]))
    corr_mout = np.zeros((K, y.shape[1]))
    y_acum = np.zeros((1,y.shape[1]))
    y_acum_hat = np.zeros((1,y.shape[1]))
    idx_acum = np.zeros((1,))
    results = dict()
    correlation_matrices = dict()
    rmse_raters = dict()

    # SVR
    base_model = SVR(kernel = 'linear')
    model = RegressorChain(base_estimator = base_model, order = [0,1,2,3,4,5])
    # model = MultiOutputRegressor(base_model)

    for k, (train_index, test_index) in enumerate(kf.split(y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(test_index)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        # print(len(model.estimators_))
        y_hat = np.abs(model.predict(X_test))
        

        if type(sev) is np.ndarray:
          sev_test,rou_test,bre_test = sev[test_index],rou[test_index],bre[test_index]
          srt_test,pit_test,lou_test = srt[test_index],pit[test_index],lou[test_index]
          
            
          R_sev[k] = label_correlation(sev_test,y_hat[:,0])
          R_rou[k] = label_correlation(rou_test,y_hat[:,1])
          R_bre[k] = label_correlation(bre_test,y_hat[:,2])
          R_srt[k] = label_correlation(srt_test,y_hat[:,3])
          R_pit[k] = label_correlation(pit_test,y_hat[:,4])
          R_lou[k] = label_correlation(lou_test,y_hat[:,5])

          RMSE_sev[k] = rmse_between_raters(sev_test,y_hat[:,0])
          RMSE_rou[k] = rmse_between_raters(rou_test,y_hat[:,1])
          RMSE_bre[k] = rmse_between_raters(bre_test,y_hat[:,2])
          RMSE_srt[k] = rmse_between_raters(srt_test,y_hat[:,3])
          RMSE_pit[k] = rmse_between_raters(pit_test,y_hat[:,4])
          RMSE_lou[k] = rmse_between_raters(lou_test,y_hat[:,5])
          

        y_acum = np.append(y_acum, y_test, axis = 0)
        y_acum_hat = np.append(y_acum_hat,y_hat, axis = 0)
        idx_acum = np.append(idx_acum, test_index)
        
        # r2mout[k] = r2_score(y_test, y_hat, multioutput = 'raw_values')
        mse[k] = np.sqrt(mean_squared_error(y_test, y_hat))
        mout[k] = np.sqrt(mean_squared_error(y_test, y_hat, multioutput = 'raw_values'))
        aux = np.corrcoef(y_test, y_hat, rowvar = False)
        r2mout[k] = [aux[0,6], aux[1,7], aux[2,8], aux[3,9], aux[4,10], aux[5,11]]
        
    mean_mse = np.mean(mse)
    std_mse = np.std(mse)
    mean_mout = np.mean(mout, axis = 0)
    std_mout = np.std(mout, axis = 0)
    mean_r2mout = np.mean(r2mout, axis = 0)
    std_r2mout = np.std(r2mout, axis = 0)
    
    rmse_raters['Severity'] = (np.mean(RMSE_sev, axis = 0), np.std(RMSE_sev, axis = 0))
    rmse_raters['Roughness'] = (np.mean(RMSE_rou, axis = 0), np.std(RMSE_rou, axis = 0))
    rmse_raters['Breathiness'] = (np.mean(RMSE_bre, axis = 0), np.std(RMSE_bre, axis = 0)) 
    rmse_raters['Strain']  = (np.mean(RMSE_srt, axis = 0), np.std(RMSE_srt, axis = 0)) 
    rmse_raters['Pitch']  = (np.mean(RMSE_pit, axis = 0), np.std(RMSE_pit, axis = 0))
    rmse_raters['Loudness'] =  (np.mean(RMSE_lou, axis = 0), np.std(RMSE_lou, axis = 0)) 



    y_acum = y_acum[1:]
    y_acum_hat = y_acum_hat[1:]
    idx_acum = idx_acum[1:]

    correlation_matrices['Severity'] = R_sev
    correlation_matrices['Roughness'] = R_rou
    correlation_matrices['Breathiness'] = R_bre
    correlation_matrices['Strain'] = R_srt
    correlation_matrices['Pitch'] = R_pit
    correlation_matrices['Loudness'] = R_lou

    results['overall_rmse'] = (mean_mse,std_mse)
    results['all_outputs_rmse'] = (mean_mout, std_mout)
    results['correlation_matrices'] = correlation_matrices
    results['acumulated_vectors'] = (y_acum, y_acum_hat,idx_acum)
    results['correlations_per_trait'] = (mean_r2mout, std_r2mout)
    results['rmse_raters'] = rmse_raters
    
    return results


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