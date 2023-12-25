import sys
sys.path.append('/Users/dorochka/Desktop/rec_sys_project/')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from preprocessing import preprocess_data
from graphs.graph_model import IGMC
from scores import hit_rate_at_k, ndcg_at_k
from config import config

torch.manual_seed(46911356)

# Load data
path = 'data/'
name = 'ratings.dat'
train_data, test_data, num_users, num_items = preprocess_data(path, name, mode='graph')

model = IGMC(num_users, num_items)

optimizer = torch.optim.Adam(model.parameters(), lr=config['graph_lr'], weight_decay=0)

losses = []
loss_fn = torch.nn.BCELoss()
for epoch in range(1, config['graph_epochs']+1):
  model.train()
  train_loss_all = 0
  
  for train_batch in train_data:
    optimizer.zero_grad()

    y_pred = model(train_batch)

    y_true = (train_batch.edge_attr-4).float().squeeze()

    train_loss = loss_fn(y_pred, y_true)
    losses.append(train_loss.item())
    train_loss.backward()
    train_loss_all += train_loss.item()
    optimizer.step()

  train_loss_all /= len(train_data)
  if epoch % config['graph_lr_decay_step'] == 0:
    for param_group in optimizer.param_groups:
      param_group['lr'] = param_group['lr'] / config['graph_lr_decay_value']
  print('epoch', epoch,' \ttrain loss', train_loss_all)


model.eval()
test_loss = 0.0
for i, test_batch in enumerate(test_data):
    with torch.no_grad():
      y_pred = model(test_batch)
      
    y_true = (test_batch.edge_attr-4).float().squeeze(1)
    test_loss += loss_fn(y_pred, y_true)
    if i == len(test_data) - 1:
      k = config['top_k']
      y_pred = y_pred.unsqueeze(1)
      y_true = y_true.unsqueeze(1)
      print(y_pred[:100, :].T)
      print(y_true[:100, :].T)
      hit_rate_score = hit_rate_at_k(y_true, y_pred, k)
      ndcg_score = ndcg_at_k(y_true, y_pred, k)
mse_loss = test_loss.item() / len(test_data)


# The hit rate of 0.7 (70%) and NDCG (Normalized Discounted Cumulative Gain) of 0.6371 are metrics to evaluate 
# the performance of your recommendation system. 
# A hit rate of 70% means that 70% of your top-K recommendations are relevant or 'hits'. 
# NDCG is a measure of ranking quality, taking into account the position of the relevant items; 
# a value of 0.6371 is quite reasonable, depending on the complexity of your task and the baseline performance.

# The test loss of 0.2388 is another indicator of your model's performance on unseen data. 
# It's relatively close to your training loss, which suggests that your model is not severely overfitting.

print(f'hitrate: {hit_rate_score}')
print(f'NDCG: {ndcg_score}')
print(f'test loss: {mse_loss}')

plt.plot(losses)
plt.title(f'train loss. last loss: {losses[-1]:.3f}')
plt.show()