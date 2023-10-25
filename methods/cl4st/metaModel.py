import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
import torch.nn.functional as F

class LinearCustom(nn.Module):

    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        """process:
        N, F -> N, 1, F
        (N, 1, F) * (N, in_dim, out_dim) = N, 1, out_dim

        Args:
            inputs (_type_): N, F
            parameters (_type_): N, in_dim, out_dim

        Returns:
            _type_: _description_
        """        
        weights, biases = parameters[0], parameters[1]
        inputs = torch.unsqueeze(inputs, dim = 1)
        
        weights = weights.transpose(1, 0)
        # print('weights: {}, biases: {}, inputs: {}'.format(weights.shape, biases.shape, inputs.shape))
        ret = F.linear(inputs, weights, biases)
        # ret = torch.matmul(inputs, weights) + biases
        ret = torch.squeeze(ret, dim = 1)
        return ret

class GINConvCustom(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, nn_params = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out, nn_params)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
        else:
            print('Using FC')
            self.weights = nn.Parameter(torch.rand( self.input_dim, self.output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand( self.output_dim), requires_grad=True)

    def forward(self, x, memory=None):
        # B, N, F
        if self.dynamic:
            weights = self.weight_generator(memory).view(x.shape[0], self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], self.output_dim)
        else:
            weights = torch.unsqueeze(self.weights, dim = 0).repeat(x.shape[0], 1, 1)
            biases = torch.unsqueeze(self.biases, dim = 0).repeat(x.shape[0], 1)
        return weights, biases