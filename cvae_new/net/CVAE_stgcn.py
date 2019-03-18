import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import *
from net.utils.graph import Graph
from utils.common import *


class CVAE(nn.Module):

    def __init__(self, in_channels, T, V, n_z, num_classes, graph_args,
                 edge_importance_weighting=False, **kwargs):

        super().__init__()

        self.T = T
        self.V = V
        self.n_z = n_z
        self.encoder = Encoder(in_channels+num_classes, n_z, graph_args, edge_importance_weighting)
        self.decoder = Decoder(in_channels, n_z+num_classes, graph_args, edge_importance_weighting)
        # self.encoder = Encoder(in_channels, n_z, graph_args, edge_importance_weighting)
        # self.decoder = Decoder(in_channels, n_z, graph_args, edge_importance_weighting)

    def forward(self, x, lenc, ldec):

        batch_size = x.size(0)

        mean, lsig = self.encoder(x, lenc)

        sig = torch.exp(0.5 * lsig)
        eps = to_var(torch.randn([batch_size, self.n_z]))
        z = eps * sig + mean

        recon_x = self.decoder(z, ldec, self.T, self.V)

        return recon_x, mean, lsig, z

    def inference(self, n=1, ldec=None):

        batch_size = n
        z = to_var(torch.randn([batch_size, self.n_z]))

        recon_x = self.decoder(z, ldec)

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

    def __init__(self, in_channels, n_z, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=75, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.encoder = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            st_gcn(32, 32, kernel_size, 1, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.encoder
            ])
        else:
            self.edge_importance = [1] * len(self.encoder)

        # fcn for encoding
        self.z_mean = nn.Conv2d(32, n_z, kernel_size=1)
        self.z_lsig = nn.Conv2d(32, n_z, kernel_size=1)

    def forward(self, x, l):

        # concat
        x = torch.cat((x, l), dim=1)

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.encoder, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

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

    def __init__(self, in_channels, n_z, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=75, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.fcn = nn.ConvTranspose2d(n_z, 32, kernel_size=1)

        self.decoder = nn.ModuleList((
            st_gctn(32, 32, kernel_size, 1, **kwargs),
            # st_gctn(32, 32, kernel_size, 1, **kwargs),
            # st_gctn(32, 32, kernel_size, 1, **kwargs),
            # st_gctn(32, 32, kernel_size, 1, **kwargs),
            st_gctn(32, 64, kernel_size, 1, **kwargs),
            # st_gctn(64, 64, kernel_size, 1, **kwargs),
            # st_gctn(64, 64, kernel_size, 1, **kwargs),
            # st_gctn(64, 64, kernel_size, 1, **kwargs),
            # st_gctn(64, 64, kernel_size, 1, **kwargs),
            st_gctn(64, in_channels, kernel_size, 1, ** kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.decoder
            ])
        else:
            self.edge_importance = [1] * len(self.decoder)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.out = nn.Sigmoid()

    def forward(self, z, l, T, V):

        N = z.size()[0]
        # concat
        z = torch.cat((z, l), dim=1)

        # reshape
        z = z.view(N, z.size()[1], 1, 1)

        # forward
        z = self.fcn(z)
        z = z.repeat([1, 1, T, V])
        # x = z.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        #
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.decoder, self.edge_importance):
            z, _ = gcn(z, self.A * importance)
        z = torch.unsqueeze(z, 4)

        # data normalization
        N, C, T, V, M = z.size()
        z = z.permute(0, 4, 3, 1, 2).contiguous()
        z = z.view(N * M, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, M, V, C, T)
        z = z.permute(0, 3, 4, 2, 1).contiguous()
        # z = self.out(z)

        return z


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class st_gctn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gctn = ConvTransposeTemporalGraphical(in_channels, out_channels,
                                                   kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gctn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
