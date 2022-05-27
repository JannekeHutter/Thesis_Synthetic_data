import pandas as pd
import numpy as np
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
from scipy.stats import bernoulli
import os
import random
from random import sample
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
plt.switch_backend('agg')
tfd = tf.contrib.distributions
import utils.process as process
import json
import utils.params as params
import seaborn as sns; sns.set(style="ticks", color_codes=True)


def data_prep(data, cat_cols, num_cols, discr_cols, target, rs=42):
    '''
    INPUT:
    data = DataFrame with all ints
    cat_cols = list of categorical columns
    num_cols = list of numerical columns
    discr_cols = list of continuous-discrete cols (eg month, day, salary which is continuous but discretized); 
    discr_cols is subset of num_cols & noise will be added to it to help training
    rs = random seed for train/test split 

    OUTPUT: all vars needed to train VAEM:
    dic_var_type: indicates which cols are categorical (=1) & numerical (=0)
    cat_dims: #categories for each categorical var;
    DIM_FLT: #numerical vars
    Data_train_compressed: train data, with reordered cols: [cat, flt]
    Data_train_decompressed: normalized train data, each categorical col replaced by one-hot encoded cols (1 for each category in cat col)
    list_discrete_compressed: index of (continuous-)discrete vars in (reordered) compressed db
    list discrete: index of (continuous-)discrete vars in decompressed db (== with more cols)
    __noisy__: same data but with noise added to (continuous-) discrete vars for training
    mask__: mtx of same shape as df, all 1s (0 would indicate missing value - not the case)
    records_d: for each vars in list_discrete: list of unique values of that var (normalized) -> for noise addition
    '''
    # make lists (np arrays) of categorical, numerical & discrete columns:
    list_cat = np.array([data.columns.get_loc(col) for col in cat_cols])
    list_flt = np.array([data.columns.get_loc(col) for col in num_cols])
    list_discrete = np.array([data.columns.get_loc(col) for col in discr_cols])

    # for scaling of data -> can be changed to 0.7 & 0.3 for example
    max_Data = 1 
    min_Data = 0
    
    # convert df to numpy array with all floats:
    Data = ((data.values).astype(float))[0:,:]
    
    # list_discrete_in_flt: index of (continuous-)discrete vars in numerical cols list
    list_discrete_in_flt = (np.in1d(list_flt, list_discrete).nonzero()[0])
    list_discrete_compressed = list_discrete_in_flt + len(list_cat)
    if len(list_flt)>0 and len(list_cat)>0:
        list_var = np.concatenate((list_cat,list_flt))
    elif len(list_flt)>0:
        list_var = list_flt
    else:
        list_var = list_cat
    Data_sub = Data[:,list_var]
    dic_var_type = np.zeros(Data_sub.shape[1])
    # the first len(list_cat) cols are categorical so label them 1:
    dic_var_type[0:len(list_cat)] = 1

    # assume no missing values in raw data (all ones):
    Mask = np.ones(Data_sub.shape) # Mask = matrix of 1s of size datset
    # Normalize/squash the data matrix: each col is normalized to [0,1]
    Data_std = (Data_sub - Data_sub.min(axis=0)) / (Data_sub.max(axis=0) - Data_sub.min(axis=0))
    # scaling_factor = 1 if max_Data=1, min_Data=0 so ignore
    #scaling_factor = (Data_sub.max(axis=0) - Data_sub.min(axis=0))/(max_Data - min_Data)
    # Data_sub = normalized data mtx with sorted columns (first categorical, then numerical)
    Data_sub = Data_std * (max_Data - min_Data) + min_Data
    
    # "decompress" categorical data into one-hot representation -> get 1 col for each category of categorical col:
    Data_cat = Data[:,list_cat].copy() # put all categorical cols into df
    Data_flt = Data[:,list_flt].copy() # put all numerical cols into df
    # Data_compressed = Data array with reordered cols: [cat, flt]
    Data_compressed = np.concatenate((Data_cat,Data_flt),axis = 1)
    Data_decompressed, Mask_decompressed, cat_dims, DIM_FLT = process.data_preprocess(Data_sub,Mask,dic_var_type)
    
    # extract new index of target y
    new_idx_target = cat_cols.index(target)
    # use y to take stratified train_test_split:  
    y = Data_compressed[:,new_idx_target] #Data_compressed[:,-1]
    Data_compressed = np.delete(Data_compressed, new_idx_target, axis=1) #Data_compressed[:, :-1] # Data_compressed becomes only input data X (without y)
    # set 20% apart for test set, taking a stratified sample based on y:
    Data_train_decompressed, Data_test_decompressed, mask_train_decompressed, mask_test_decompressed,mask_train_compressed, mask_test_compressed,Data_train_compressed, Data_test_compressed, y_train, y_test = train_test_split(Data_decompressed, Mask_decompressed,Mask,Data_compressed,y, test_size=0.2, stratify=y, random_state=rs)
    # reappend target y column to (compressed) train & test sets:
    Data_train_compressed = np.column_stack((Data_train_compressed[:, :-DIM_FLT], y_train, Data_train_compressed[:, -DIM_FLT:]))
    #Data_train_compressed = np.column_stack((Data_train_compressed, y_train))
    Data_test_compressed = np.column_stack((Data_test_compressed[:,:-DIM_FLT], y_test, Data_test_compressed[:, -DIM_FLT:]))
    #Data_test_compressed = np.column_stack((Data_test_compressed, y_test))

    list_discrete = list_discrete_in_flt + (cat_dims.sum()).astype(int)
    
    Data_decompressed = np.concatenate((Data_train_decompressed, Data_test_decompressed), axis=0)
    Data_train_orig = Data_train_decompressed.copy()
    Data_test_orig = Data_test_decompressed.copy()

    # Add some noise to continuous-discrete variables to help training; disable by setting noise_ratio=0
    Data_noisy_decompressed,records_d, intervals_d = process.noisy_transform(Data_decompressed, list_discrete, noise_ratio = 0.99)
    ## noise_record = mtrx of size Data, with only added noise or 0 if no noise added:
    noise_record = Data_noisy_decompressed - Data_decompressed
    Data_train_noisy_decompressed = Data_noisy_decompressed[0:Data_train_decompressed.shape[0],:]
    Data_test_noisy_decompressed = Data_noisy_decompressed[Data_train_decompressed.shape[0]:,:]
    
    return Data_train_compressed, Data_train_decompressed, Data_train_noisy_decompressed, mask_train_decompressed, Data_test_decompressed, mask_test_compressed, mask_test_decompressed, cat_dims, DIM_FLT, dic_var_type, records_d, list_discrete, list_discrete_compressed