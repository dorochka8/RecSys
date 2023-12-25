import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 

path = 'data/'
name = 'ratings.dat'
data = pd.read_csv(os.path.join(path, name), 
                    sep='::', header=None, names=['user', 'item', 'rating'], 
                    usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int8})

fig = plt.figure()
ax = data.rating.value_counts(normalize=True).sort_index().plot.bar()
values = ax.get_yticks()
ax.set_yticklabels(['{:.2f}'.format(x) for x in values])
plt.xlabel('rating')
plt.ylabel('share of ratings')
plt.title('movielens 1M dataset ratings distribution')
plt.show()