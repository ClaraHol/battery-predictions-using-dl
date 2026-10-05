import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq

def load_data():
    return

def train_oracle(metadata_list_train, 
                 feature_data_list_train, 
                 metadata_list_test, 
                 feature_data_list_test, 
                 alpha_f_sbi,
                 C_f_ope,
                 epsilon_f_ope,
                 num_features,
                 l1_ratio
    ):
    model_parameters_train = []
    for i in len(metadata_list_train):
        feature_data = feature_data_list_train[i]
        features = feature_data[:, num_features]
        
    return

def validate_oracle():
    return
