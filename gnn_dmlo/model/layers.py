from typing import Tuple, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    Optional,
    OptPairTensor,
    Size,
)


class GeneralConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: Optional[int],
        in_edge_channels: int = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels

        # message for undirected edges
        self.lin_msg = Linear(in_channels, out_channels, bias=bias)
        self.lin_msg_i = Linear(in_channels, out_channels, bias=bias)

        # skip linear
        self.lin_self = Linear(in_channels, out_channels, bias=bias)

        if self.in_edge_channels is not None:
            self.lin_edge = Linear(in_edge_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_msg.reset_parameters()
        if hasattr(self.lin_self, "reset_parameters"):
            self.lin_self.reset_parameters()
        if self.in_edge_channels is not None:
            self.lin_edge.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor = None, size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_self = x[1]

        # propagate_type: (x: OptPairTensor, edge_attr: Tensor)
        out = self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
        out = out + self.lin_self(x_self)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)

        if edge_attr is not None:
            x_j = x_j + self.lin_edge(edge_attr)

        return x_j
