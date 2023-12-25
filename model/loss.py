import torch as K
import torch
import functools
from torch import nn
import torch.nn.functional as F
import numpy as np
mseLoss = nn.MSELoss()
def std_mae(std=1):
    def mae(y_pred,y_true):
        return K.mean(K.abs(y_pred - y_true)) * std
    return mae


def std_mse(std=1):
    def mse(y_pred,y_true):
        return mseLoss(y_true, y_pred)
    return mse

def std_rmse(std=1):
    def rmse(y_pred,y_true):
        return K.sqrt(mseLoss(y_true, y_pred)) * std
    return rmse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def r2_s(y_pred,y_true):
    return r2_score(y_true + 1e-7, y_pred + 1e-7)

def std_r2(std=1):
    def r2(y_pred,y_true):
        ss_res = K.sum(K.square((y_true - y_pred) * std))
        ss_tot = K.sum(K.square((y_true - K.mean(y_true) * std)))
        return 1 - ss_res / (ss_tot + 1e-7)
    return r2
