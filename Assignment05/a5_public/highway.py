#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch

class Highway(torch.nn.Module):
    def __init__(self, emb_size):
        super(Highway, self).__init__()
        self.w_proj = torch.nn.Linear(in_features=emb_size, out_features=emb_size)
        self.w_gate = torch.nn.Linear(in_features=emb_size, out_features=emb_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x_convout):
        # x_convout : (max_sen_length, batch_size, emb_size)
        x_proj = self.relu( self.w_proj(x_convout) )
        x_gate = self.sigmoid( self.w_gate(x_convout))
        x_highway = x_gate*x_proj + ( torch.ones( x_gate.size() ) - x_gate ) * x_convout
        x_wordemb = self.dropout(x_highway)
        return x_wordemb
### END YOUR CODE 

