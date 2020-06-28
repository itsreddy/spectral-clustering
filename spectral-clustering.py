import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
import io, csv, os
import numpy as np
from collections import defaultdict
import pickle
import scipy
import sklearn.metrics
from sklearn.utils import shuffle

import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes

base_path = os.getcwd()

def save_obj(obj, name ):
    with open( base_path + '/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(base_path + '/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def split_df(raw_df, split):
    split_index = round(split * raw_df.shape[0])
    train_df = raw_df[split_index: ]
    validate_df = raw_df[: split_index]
    return train_df, validate_df

def create_dict(id_list):
    id_dict = {}
    cursor = 0
    for val in id_list:
        if val not in id_dict:
            id_dict[val] = cursor
            cursor += 1
    print(len(id_dict))
    return id_dict





os.makedirs(base_path + '/obj/', exist_ok=True)
raw_df = pd.read_csv( base_path + "train.csv")
raw_df = shuffle(raw_df)
raw_df = raw_df.reset_index(drop=True)

train_df, validate_df = split_df(raw_df, 0.8)

train_customers = train_df['customer-id']
train_cid_sorted = train_customers.sort_values()
train_movies = train_df['movie-id']
train_mid_sorted = train_movies.sort_values()

train_cid = create_dict(train_cid_sorted)
train_mid = create_dict(train_mid_sorted)

