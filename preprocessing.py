import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch 

from collections import defaultdict
from torch_geometric.data import Data


def preprocess_data(data_dir, train_rating, mode='graph', num_train_subgraphs=25, num_test_subgraphs=34):
    data = pd.read_csv(os.path.join(data_dir, train_rating), 
                       sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], 
                       dtype={0: np.int32, 1: np.int32, 2: np.int8},
                       engine='python')
    data = data[data['rating'] > 3]
    train_data = data.sample(frac=0.8, random_state=46911356)
    test_data = data.drop(train_data.index)

    if mode == 'graph':
        num_users = len(data['user'].unique())
        num_items = len(data['item'].unique())

        global_user_mapping = {user_id: idx for idx, user_id in enumerate(data['user'].unique())}
        global_movie_mapping = {movie_id: idx for idx, movie_id in enumerate(data['item'].unique(), start = len(global_user_mapping))}

        train_subgraphs = do_subgraphs(train_data, global_user_mapping, global_movie_mapping, num_train_subgraphs)
        test_subgraphs = do_subgraphs(test_data, global_user_mapping, global_movie_mapping, num_test_subgraphs)

        return train_subgraphs, test_subgraphs, num_users, num_items
    
    elif mode == 'svdpp':
        return (train_data, test_data)
    
    elif mode in 'rl':
        test_data = test_data[['user', 'item']].values.tolist()
        train_data = train_data[['user', 'item']].values.tolist()
        user_num = data['user'].max() + 1
        item_num = data['item'].max() + 1

        # defaultdict: defines values for unexistent (yet) keys ?/????//?/?
        train_mat = defaultdict(int)
        test_mat = defaultdict(int)
        
        # Build matrices where a row is a user and column is a film
        for user, item in train_data:
            train_mat[user, item] = 1.0
        for user, item in test_data:
            test_mat[user, item] = 1.0
        
        # dok_matrix: dict of keys based sparse matrix
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)

        test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(test_matrix, test_mat)
        appropriate_users = np.arange(user_num).reshape(-1, 1)[(train_matrix.sum(1) >= 20)] 
        
        return (train_data, train_matrix, test_data, test_matrix, user_num, item_num, appropriate_users)

    raise ValueError(f"expected mode 'rl', 'graph', 'svdpp', got '{mode}'")


def mapping(path, name, cols, index_col=None):
    dataread = pd.read_csv(os.path.join(path, name), 
                            sep='::', header=None, names=cols, 
                            engine='python', index_col=index_col)
    mapping = {idx: i for i, idx in enumerate(dataread.index)}
    return mapping


def do_subgraphs(train_data, global_user_mapping, global_movie_mapping, num_subgraphs):
    subgraphs = []
    users_per_subgraph = len(global_user_mapping) // num_subgraphs
    for i in range(num_subgraphs):
        start_idx = i * users_per_subgraph
        end_idx = start_idx + users_per_subgraph if i < num_subgraphs - 1 else len(global_user_mapping)
        selected_users = list(global_user_mapping.keys())[start_idx:end_idx]

        subgraph_data = train_data[train_data['user'].isin(selected_users)]            
        src = [global_user_mapping[idx] for idx in subgraph_data['user']]
        dst = [global_movie_mapping[idx] for idx in subgraph_data['item']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(subgraph_data['rating'].values, dtype=torch.float).unsqueeze(1)
        num_nodes = len(subgraph_data['user'].unique()) + len(subgraph_data['item'].unique())

        subgraph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        subgraphs.append(subgraph)
    return subgraphs



path = 'data/'
name = 'ratings.dat'
train_data, test_data, num_users, num_items = preprocess_data(path, name, mode='graph')