#!/usr/bin/env python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation, bn_momentum):
        super(ResidualBlock, self).__init__()

        padding = (kernel_size - 1) * dilation // 2
        hidden_dim2 = hidden_dim*2

        self.conv1 = nn.Conv1d(in_channels=hidden_dim,
                               out_channels=hidden_dim2,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation=dilation)

        self.conv1_0 = nn.Conv1d(in_channels=hidden_dim,
                                 out_channels=hidden_dim2,
                                 kernel_size=3,
                                 padding=1,
                                 dilation=1)

        self.norm1 = nn.BatchNorm1d(hidden_dim2, momentum=bn_momentum)

        self.conv2 = nn.Conv1d(in_channels=hidden_dim2,
                               out_channels=hidden_dim,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.conv2_0 = nn.Conv1d(in_channels=hidden_dim2,
                                 out_channels=hidden_dim,
                                 kernel_size=3,
                                 padding=1,
                                 dilation=1)

        self.norm2 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)

    def forward(self, X0):
        X = self.conv1(X0) + self.conv1_0(X0)
        X = torch.relu(X)
        X1 = self.norm1(X)
        X = self.conv2(X1) + self.conv2_0(X1)
        X = torch.relu(X)
        X2 = self.norm2(X)
        Xout = X2 + X0
        return Xout


class Network(nn.Module):
    def __init__(self, para):
        super(Network, self).__init__()
        self.input_dim = para['input_dim']
        self.output_dim = para['output_dim']
        self.bn_momentum = para['bn_momentum']

        self.hidden_dim = para['hidden_dim']
        self.kernel_size = para['kernel_size']
        self.dilation = para['dilation']
        self.num_layers = para['num_layers']
        self.padding = (self.kernel_size - 1) * self.dilation // 2

        self.conv_input = nn.Conv1d(in_channels=self.input_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=5, padding=2, dilation=1)

        self.norm_input = nn.BatchNorm1d(
            self.hidden_dim, momentum=self.bn_momentum)

        resblocks = []
        for i in range(0, self.num_layers):
            resblocks += [ResidualBlock(self.hidden_dim, self.kernel_size,
                                        self.dilation, self.bn_momentum)]
        self.resblocks = nn.ModuleList(resblocks)

        self.conv_o = nn.Conv1d(in_channels=self.hidden_dim,
                                out_channels=self.output_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                dilation=self.dilation)

    def forward(self, X0):
        X = X0.permute(0, 2, 1)

        X = self.conv_input(X)
        X = torch.relu(X)
        X = self.norm_input(X)
        for i in range(0, self.num_layers):
            res_block = self.resblocks[i]
            X = res_block(X)

        out = self.conv_o(X)
        out = out.permute(0, 2, 1)

        return out
