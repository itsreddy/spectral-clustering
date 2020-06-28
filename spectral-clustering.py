import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
import io, csv, os, sys
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
sys.path.append(base_path)

import save_utils

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

def make_matrix(train_df, train_cid, train_mid):
    # Make movie-by-user matrix to hold ratings
    train_ratings = np.zeros((train_cust_n, train_movies_n), np.short)
    for i in range(len(train_df)):
        current = train_df.iloc[i]
        customer_idx = train_cid.get(current[1])
        movie_idx = train_mid.get(current[0])
        train_ratings[customer_idx, movie_idx] = current[2]
    return train_ratings

def preprocess_matrix(train_ratings):
    train_processed_mov = np.zeros(train_ratings.shape)
    for i in range(train_ratings.shape[1]):
        rating_movie = train_ratings[:, i]
        train_processed_mov[:, i] = rating_movie - np.mean(rating_movie)
    return train_processed_mov




class options:
    def __init__(self):
        self.save_objs = False
        self.load_objs = False
        self.split_dataset = True

opt = options()

if opt.save_objs or opt.load_objs:
    os.makedirs(base_path + '/obj/', exist_ok=True)
    pre = base_path + '/obj/'

raw_df = pd.read_csv( base_path + "train.csv")
raw_df = shuffle(raw_df)
raw_df = raw_df.reset_index(drop=True)

if opt.split_dataset:
    train_df, validate_df = split_df(raw_df, 0.8)
else:
    train_df, validate_df = split_df(raw_df, 0.0)

train_customers = train_df['customer-id']
train_cid_sorted = train_customers.sort_values()
train_movies = train_df['movie-id']
train_mid_sorted = train_movies.sort_values()

train_cid = create_dict(train_cid_sorted)
train_mid = create_dict(train_mid_sorted)

train_movies_n = len(train_mid)
train_cust_n = len(train_cid)
print(train_movies_n, train_cust_n)

train_ratings = make_matrix(train_df, train_cid, train_mid)

if opt.save_objs:
    save_utils.save_obj(train_ratings, pre+'train_ratings')
if opt.load_objs:
    train_ratings = save_utils.load_obj(pre+'train_ratings')



