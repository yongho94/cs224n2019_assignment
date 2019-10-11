#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
from torch import nn
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, emb_size):
        # input : (max_sentence_length, batch_size, emb_dim, max_word_length) => (??, ?, 50, 21)
        # conv_kernel : 후, (??, ?,
        super(CNN, self).__init__()
        self.kernel_size = (5, 3)
        self.emb_size = emb_size
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=emb_size, kernel_size=self.kernel_size)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(17, stride=17)

    def forward(self, x_reshaped):
        x_conv = self.conv1d(x_reshaped) # (max_sen_length, batch_size, emb_dim, max_word_length) => (max_sen_length, batch_size, 17, emb_size )
        x_relu = self.relu(x_conv)
        x_relu = x_relu.view( list(x_relu.size()[:2]) + [-1])
        assert x_relu.size()[-1] % 17 == 0
        x_convout = self.max_pool(x_relu) # (max_sen_length, batch_size, emb_size)
        return x_convout
### END YOUR CODE