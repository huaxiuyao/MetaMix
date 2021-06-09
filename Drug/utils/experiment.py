#!/usr/bin/python

from sklearn.ensemble import RandomForestRegressor
import numpy as np



def pearson_score(y_true, y_pred):
    try:
        assert y_true.shape[-1] == y_pred.shape[-1] 
    except:
        y_pred = np.squeeze(y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2