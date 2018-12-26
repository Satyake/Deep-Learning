#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:25:26 2018

@author: satyake
"""

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W = self.W+(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b =self.b+torch.sum((v0 - vk))
        self.a =self.a+torch.sum((ph0 - phk))
        