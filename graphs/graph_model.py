import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
class IGMC(nn.Module):
  def __init__(self, num_users, num_items):
    super().__init__()
    self.num_users = num_users
    self.num_items = num_items
    total_features = num_users + num_items 

    self.rel_graph_convs = torch.nn.ModuleList()
    self.rel_graph_convs.append(GCNConv(in_channels=total_features, out_channels=32, num_relations=2, num_bases=4))
    self.rel_graph_convs.append(GCNConv(in_channels=32, out_channels=64, num_relations=2, num_bases=4))
    self.rel_graph_convs.append(GCNConv(in_channels=64, out_channels=64, num_relations=2, num_bases=4))
    self.rel_graph_convs.append(GCNConv(in_channels=64, out_channels=32, num_relations=2, num_bases=4))
    
    self.attention_layer = GATConv(in_channels=32, out_channels=512)
    self.linear_layer1 = nn.Linear(1024, 128)
    self.linear_layer2 = nn.Linear(128, 1)


  def reset_parameters(self):
    self.attention_layer.reset_parameters()
    self.linear_layer1.reset_parameters()
    self.linear_layer2.reset_parameters()
    for i in self.rel_graph_convs:
      i.reset_parameters()


  def forward(self, data):
    # If you're not using additional features for users and items, 
    # this one-hot encoding approach is a common way to represent 
    # nodes in graph-based recommendation systems. 
    # Each row of x then becomes the feature vector for a node,
    #  where num_users + num_items is the total number of nodes.
    x = torch.eye(self.num_users + self.num_items)
    edge_index = data.edge_index

    for conv in self.rel_graph_convs:
        x = conv(x, edge_index)
        x = torch.tanh(x)
        
    x = self.attention_layer(x, edge_index)
    x = F.relu(x)

    # Select the features for source and target nodes
    source_features = x[edge_index[0]]
    target_features = x[edge_index[1]]

    # # Concatenate or otherwise combine these features
    combined_features = torch.cat([source_features, target_features], dim=1)

    # # Pass through linear layers
    out = self.linear_layer1(combined_features)
    out = F.relu(out)
    out = F.dropout(out, p=0.5)
    out = self.linear_layer2(out)
    return torch.sigmoid(out).squeeze()
