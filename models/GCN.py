import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv,SGConv,GINConv
from dgl.nn import GATv2Conv


class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
        return h_list, h


def GCN5_64(dataset):
    return GCN(num_layers=5, hidden_dim=64)
    
def GCN5_32(dataset):
    return GCN(num_layers=5, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, graph_pooling_type='sum')

def GCN3_64(dataset):
    return GCN(num_layers=3, input_dim=dataset.dim_nfeats, hidden_dim=64,
               output_dim=dataset.gclasses, final_dropout=0.5, graph_pooling_type='sum')
    
def GCN3_32(dataset):
    return GCN(num_layers=3, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, graph_pooling_type='sum')
 
def GCN2_64(dataset):
    return GCN(num_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=64,
               output_dim=dataset.gclasses, final_dropout=0.5, graph_pooling_type='sum')    
    
def GCN2_32(dataset):
    return GCN(num_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, graph_pooling_type='sum')