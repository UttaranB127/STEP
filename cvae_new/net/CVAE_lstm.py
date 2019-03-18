import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import *


class CVAE(nn.Module):

    def __init__(self, in_channels, T, n_z, num_classes):

        super().__init__()

        self.T = T
        self.n_z = n_z
        self.encoder = Encoder(T, in_channels+num_classes, n_z)
        self.decoder = Decoder(T, in_channels, n_z+num_classes)
        # self.encoder = Encoder(in_channels, n_z, graph_args, edge_importance_weighting)
        # self.decoder = Decoder(in_channels, n_z, graph_args, edge_importance_weighting)

    def forward(self, x, lenc, ldec):

        batch_size = x.size(0)

        mean, lsig = self.encoder(x, lenc)

        sig = torch.exp(0.5 * lsig)
        eps = to_var(torch.randn([batch_size, self.n_z]))
        z = eps * sig + mean

        recon_x = self.decoder(z, ldec, self.T)

        return recon_x, mean, lsig, z

    def inference(self, n=1, ldec=None):

        batch_size = n
        z = to_var(torch.randn([batch_size, self.n_z]))

        recon_x = self.decoder(z, ldec, self.T)

        return recon_x


class Encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, T, in_channels, n_z):
        super().__init__()

        self.data_bn = nn.BatchNorm1d(in_channels)
        # self.lstm = nn.LSTM(in_channels, 64, 3)
        self.lstm = nn.ModuleList((
            nn.LSTM(in_channels, 64, 3),
            nn.LSTM(64, 32, 3)
        ))

        # fcn for encoding
        self.z_mean = nn.Conv2d(T*32, n_z, kernel_size=1)
        self.z_lsig = nn.Conv2d(T*32, n_z, kernel_size=1)

    def forward(self, x, l):

        # concat
        x = torch.cat((x, l), dim=2)

        # data normalization
        x = x.permute(0, 2, 1).contiguous()
        x = self.data_bn(x)
        x = x.permute(2, 0, 1).contiguous()

        # forward
        for layer in self.lstm:
            x, _ = layer(x)
        # x = x[-1, :, :].view(x.shape[1], x.shape[2], 1, 1)
        x = x.view(x.shape[1], x.shape[0]*x.shape[2], 1, 1)

        # prediction
        mean = self.z_mean(x)
        mean = mean.view(mean.size(0), -1)
        lsig = self.z_lsig(x)
        lsig = lsig.view(lsig.size(0), -1)

        return mean, lsig


class Decoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, T, in_channels, n_z):
        super().__init__()

        # build networks
        self.fcn = nn.ConvTranspose2d(n_z, T*32, kernel_size=1)

        self.lstm = nn.ModuleList((
            nn.LSTM(32, 64, 3),
            nn.LSTM(64, in_channels, 3)
        ))

        self.data_bn = nn.BatchNorm1d(in_channels)
        self.out = nn.Sigmoid()

    def forward(self, z, l, T):

        N = z.size()[0]
        # concat
        z = torch.cat((z, l), dim=1)

        # reshape
        z = z.view(N, z.size()[1], 1, 1)

        # forward
        z = self.fcn(z)
        # z = z.view(z.shape[0], z.shape[1], 1)
        # z = z.repeat([1, 1, T]).permute(2, 0, 1).contiguous()
        z = z.view(T, z.shape[0], int(z.shape[1]/T))
        # x = z.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        #
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)

        # forward
        for layer in self.lstm:
            z, _ = layer(z)

        # data normalization
        z = z.permute(1, 2, 0).contiguous()
        z = self.data_bn(z)
        z = z.permute(0, 2, 1).contiguous()
        z = self.out(z)

        return z
