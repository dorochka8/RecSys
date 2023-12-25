import sys
sys.path.append('/path/to/the/project/')

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

optimizer = torch.optim.Adam(model.parameters(), lr=config['graph_lr'], weight_decay=config['graph_weight_decay'])

losses = []
for epoch in range(1, config['graph_epochs']+1):
  model.train()
  train_loss_all = 0
  
  for train_batch in train_data:
    optimizer.zero_grad()

    y_pred = model(train_batch)
    y_true = (train_batch.edge_attr-4).float().squeeze()
    train_loss = F.mse_loss(y_pred, y_true)
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
      test_loss += F.mse_loss(y_pred, y_true)
    if i == len(test_data) - 1:
      k = config['top_k']
      hit_rate_score = hit_rate_at_k(y_true, y_pred, k)
      ndcg_score = ndcg_at_k(y_true, y_pred, k)
mse_loss = test_loss.item() / len(test_data)


print(f'hitrate: {hit_rate_score}')
print(f'NDCG: {ndcg_score}')
print(f'test loss: {mse_loss}')

plt.plot(losses)
plt.title(f'train loss. last loss: {losses[-1]:.3f}')
plt.show()