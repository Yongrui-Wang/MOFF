import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

import torch
import torch.nn as nn
import torch.nn.functional as F



msg = fn.copy_src(src='x', out='m')
def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}

def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'ft': accum}

class GCNLayer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dropout=0.,
                 agg="sum",
                 is_normalize=False,
                 residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, g):
        with g.local_scope():
            h_in = g.ndata['x']
            if self.agg == "sum":
                g.update_all(fn.copy_src('x', 'm'), reduce_sum)
            elif self.agg == "mean":
                g.update_all(fn.copy_src('x', 'm'), reduce_mean)
            h = self.linear1(g.ndata['x'])
            h = self.activation(h)
            if self.is_normalize:
                h = F.normalize(h, p=2, dim=1)
            if self.residual:
                h = h+h_in
            return h

class GATLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 h_drop=0.6,
                 a_drop=0.,
                 activation=None,
                 residual=False):
        super(GATLayer, self).__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.out_channels = out_channels
        # Linear transformation for the input features
        self.linear = nn.Linear(in_channels, out_channels * num_heads, bias=False)

        # Attention mechanism parameters
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_channels)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_channels)))
        # Dropout layer
        self.dropout = nn.Dropout(h_drop)
        self.attn_drop = nn.Dropout(a_drop)
        # LeakyReLU activation
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Initialize weights

        if residual:
            if in_channels != out_channels * num_heads:
                self.res_fc = nn.Linear(
                    in_channels, num_heads * out_channels, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        gain = nn.init.calculate_gain('relu')

        nn.init.xavier_normal_(self.linear.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g):
        with g.local_scope():
            h_in = g.ndata['x']
            src_prefix_shape = dst_prefix_shape = h_in.shape[:-1]
            h_src = h_dst = self.dropout(h_in)
            feat_src = feat_dst = self.linear(h_src).view(
                *src_prefix_shape, self.num_heads, self.out_channels)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            g.srcdata.update({'x': feat_src, 'el': el})
            g.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            # compute softmax
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))
            # message passing
            g.update_all(fn.u_mul_e('x', 'a', 'm'), fn.sum('m', 'x'))
            rst = g.dstdata['x']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.out_channels)
                rst = rst + resval

            # activation
            if self.activation:
                rst = self.activation(rst)


            return rst


class GINLayer(nn.Module):
    def __init__(self,
                 apply_func=None,
                 aggregator_type='sum',
                 init_eps=0,
                 dropout=0.,
                 learn_eps=False,
                 activation=None):
        super(GINLayer, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, g):
        _reducer = getattr(fn, self._aggregator_type)
        with g.local_scope():
            x = g.ndata['x']
            aggregate_fn = fn.copy_src('x', 'm')

            feat_src, feat_dst = expand_as_pair(x, g)
            g.srcdata['h'] = feat_src
            g.update_all(aggregate_fn, _reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + g.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst



class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)