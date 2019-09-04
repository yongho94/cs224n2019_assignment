#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
from torch import nn
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, input, batch_size, emb_size):
        self.kernel_size = 5
        self.input = input
        self.batch_size = batch_size
        self.emb_size = emb_size
### END YOUR CODE