import dgl
import torch
import torch.nn as nn

import dgl.nn as dglnn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.5, embedding=False):
        super().__init__()

        self.embedding = embedding
        if self.embedding:
            self.embedding = torch.nn.Embedding(in_feats, embedding_dim=hid_feats)
            self.conv1 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats)
        else:
            self.conv1 = dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats)
        self.conv2 = dglnn.GraphConv(in_feats=hid_feats, out_feats=out_feats)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, input):
        g = dgl.add_self_loop(graph)
        if self.embedding:
            features = self.embedding(input)
        else:
            features = input
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h
