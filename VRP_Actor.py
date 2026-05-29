import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT = True

V_UAV_MAX = 6.0
V_ADR_MAX = 2.49

class HetGatConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_channels,
                 n_heads=8, negative_slope=0.2, dropout=0.0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.Wq = nn.Linear(in_channels, out_channels, bias=False)
        self.Wk = nn.Linear(in_channels, out_channels, bias=False)
        self.Wv = nn.Linear(in_channels, out_channels, bias=False)
        self.We = nn.Linear(edge_channels, n_heads, bias=True)
        self.role_bias_all    = nn.Parameter(torch.zeros(n_heads))
        self.role_bias_pickup = nn.Parameter(torch.zeros(n_heads))
        self.role_bias_pair   = nn.Parameter(torch.zeros(n_heads))
        self.norm = nn.LayerNorm(out_channels)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, num_depots, num_nodes,
                mask, batch_size, size=None):
        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)
        n_pickup = num_nodes // 2

        xq_v = xq.view(batch_size, -1, self.out_channels)
        xk_v = xk.view(batch_size, -1, self.out_channels)
        xv_v = xv.view(batch_size, -1, self.out_channels)

        xq_p = xq_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)
        xk_p = xk_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)
        xv_p = xv_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)

        edge_attr_v = edge_attr.view(batch_size, num_depots + num_nodes, num_depots + num_nodes, -1)
        edge_attr_p = edge_attr_v[:, num_depots:num_depots + n_pickup,
                                     num_depots:num_depots + n_pickup, :].reshape(-1, edge_attr.size(-1))
        edge_attr_flat = edge_attr_v.reshape(-1, edge_attr.size(-1))

        mask_v    = mask.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        mask_p    = mask_v[:, num_depots:num_depots + n_pickup,
                              num_depots:num_depots + n_pickup].reshape(-1, 1)
        mask_flat = mask_v.reshape(-1, 1)

        ei_sub = torch.stack([
            torch.arange(batch_size * n_pickup, device=x.device).repeat_interleave(n_pickup),
            torch.arange(batch_size * n_pickup, device=x.device).view(-1, n_pickup)
                .repeat_interleave(n_pickup, dim=0).flatten()
        ])

        batch_off = (torch.arange(batch_size, device=x.device)
                     .repeat_interleave(n_pickup) * (2 * n_pickup))
        local_p   = torch.arange(n_pickup, device=x.device).repeat(batch_size)
        abs_p     = batch_off + local_p
        abs_d     = batch_off + local_p + n_pickup
        ei_pair   = torch.stack([torch.cat([abs_p, abs_d]),
                                  torch.cat([abs_d, abs_p])])

        xq_pd = xq_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)
        xk_pd = xk_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)
        xv_pd = xv_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)

        diag  = torch.arange(n_pickup, device=x.device)
        ea_p2d = (edge_attr_v[:, num_depots:num_depots + n_pickup,
                                 num_depots + n_pickup:num_depots + 2 * n_pickup, :]
                  [:, diag, diag, :].reshape(-1, edge_attr.size(-1)))
        ea_d2p = (edge_attr_v[:, num_depots + n_pickup:num_depots + 2 * n_pickup,
                                 num_depots:num_depots + n_pickup, :]
                  [:, diag, diag, :].reshape(-1, edge_attr.size(-1)))
        edge_attr_pair = torch.cat([ea_p2d, ea_d2p], dim=0)
        mask_pair      = torch.ones(2 * batch_size * n_pickup, 1, device=x.device)

        X_all  = self.propagate(edge_index, q=xq,    k=xk,    v=xv,    edge_attr=edge_attr_flat, mask=mask_flat, key='all',    size=size)
        X_pick = self.propagate(ei_sub,     q=xq_p,  k=xk_p,  v=xv_p,  edge_attr=edge_attr_p,   mask=mask_p,   key='pickup',  size=None)
        X_pair = self.propagate(ei_pair,    q=xq_pd, k=xk_pd, v=xv_pd, edge_attr=edge_attr_pair, mask=mask_pair, key='pair',   size=None)

        X_all_v  = X_all.view(batch_size, -1, self.out_channels)
        X_pick_v = X_pick.view(batch_size, n_pickup, self.out_channels)
        X_pair_v = X_pair.view(batch_size, 2 * n_pickup, self.out_channels)

        out = torch.cat([
            X_all_v[:, :num_depots, :],
            X_all_v[:, num_depots:num_depots + n_pickup, :]           + X_pick_v + X_pair_v[:, :n_pickup, :],
            X_all_v[:, num_depots + n_pickup:num_depots + num_nodes, :]            + X_pair_v[:, n_pickup:, :],
        ], dim=1).view(-1, self.out_channels)

        return self.norm(out + xv)

    def message(self, edge_index_i, q_i, k_j, v_j, size_i, edge_attr, mask, key):
        E = q_i.size(0)
        H, d = self.n_heads, self.head_dim
        q = q_i.view(E, H, d)
        k = k_j.view(E, H, d)
        v = v_j.view(E, H, d)
        alpha = (q * k).sum(-1) / self.scale
        alpha = alpha + self.We(edge_attr)
        if key == 'all':
            alpha = alpha + self.role_bias_all
        elif key == 'pickup':
            alpha = alpha + self.role_bias_pickup
        else:
            alpha = alpha + self.role_bias_pair
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(mask.expand(-1, H) == 0, -1e9)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (v * alpha.unsqueeze(-1)).reshape(E, H * d)

    def update(self, aggr_out):
        if self.training and torch.isnan(aggr_out).any():
            print('[HetGatConv] NaN in aggregation — isolated node or mask bug', flush=True)
        return torch.nan_to_num(aggr_out, nan=0.0)

class SimpleGatConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_channels,
                 n_heads=8, negative_slope=0.2, dropout=0.0):
        super().__init__(aggr='add')
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.Wq = nn.Linear(in_channels, out_channels, bias=False)
        self.Wk = nn.Linear(in_channels, out_channels, bias=False)
        self.Wv = nn.Linear(in_channels, out_channels, bias=False)
        self.We = nn.Linear(edge_channels, n_heads, bias=True)
        self.norm = nn.LayerNorm(out_channels)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, mask, batch_size, size=None):
        edge_attr_flat = edge_attr.view(-1, edge_attr.size(-1))
        mask_flat = mask.view(-1, 1)

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        out = self.propagate(edge_index, q=xq, k=xk, v=xv,
                             edge_attr=edge_attr_flat, mask=mask_flat, size=size)
        return self.norm(out + xv)

    def message(self, edge_index_i, q_i, k_j, v_j, size_i, edge_attr, mask):
        E = q_i.size(0)
        H, d = self.n_heads, self.head_dim
        q = q_i.view(E, H, d)
        k = k_j.view(E, H, d)
        v = v_j.view(E, H, d)
        alpha = (q * k).sum(-1) / self.scale
        alpha = alpha + self.We(edge_attr)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(mask.expand(-1, H) == 0, -1e9)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (v * alpha.unsqueeze(-1)).reshape(E, H * d)

    def update(self, aggr_out):
        return torch.nan_to_num(aggr_out, nan=0.0)

class SimpleGatEncoder(nn.Module):

    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=1, hidden_edge_dim=16, conv_layers=3):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim

        self.W_depot    = nn.Linear(input_node_dim, hidden_node_dim, bias=False)
        self.W_pickup   = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        self.W_edge_uav  = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.W_edge_adr  = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.ln_node     = nn.LayerNorm(hidden_node_dim)
        self.ln_edge_uav = nn.LayerNorm(hidden_edge_dim)
        self.ln_edge_adr = nn.LayerNorm(hidden_edge_dim)

        self.mode_token_uav = nn.Parameter(torch.zeros(hidden_node_dim))
        self.mode_token_adr = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.mode_token_uav, std=0.02)
        nn.init.normal_(self.mode_token_adr, std=0.02)

        self.convs_uav = nn.ModuleList([
            SimpleGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
            for _ in range(conv_layers)
        ])
        self.convs_adr = nn.ModuleList([
            SimpleGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
            for _ in range(conv_layers)
        ])

        self.ff_uav = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim), nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ff_adr = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim), nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ln_final_uav = nn.LayerNorm(hidden_node_dim)
        self.ln_final_adr = nn.LayerNorm(hidden_node_dim)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data['x'].shape[0]
        feat  = data['x']
        n_total = feat.shape[1]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = n_total - num_depots
        n_pickup  = num_nodes // 2

        depot_f         = feat[:, :num_depots, :].reshape(batch_size * num_depots, -1)
        pickup_f        = feat[:, num_depots:num_depots + n_pickup, :]
        delivery_f      = feat[:, num_depots + n_pickup:, :]
        pickup_paired   = torch.cat([pickup_f,   delivery_f], dim=-1).reshape(batch_size * n_pickup, -1)
        delivery_paired = torch.cat([delivery_f, pickup_f],   dim=-1).reshape(batch_size * n_pickup, -1)

        emb_depot    = self.W_depot(depot_f)
        emb_pickup   = self.W_pickup(pickup_paired)
        emb_delivery = self.W_delivery(delivery_paired)

        x = torch.cat([
            emb_depot.view(batch_size, num_depots, -1),
            emb_pickup.view(batch_size, n_pickup, -1),
            emb_delivery.view(batch_size, n_pickup, -1),
        ], dim=1).view(-1, self.hidden_node_dim)
        x = self.ln_node(x)

        ea_uav = data['edge_attr_uav'].view(-1, data['edge_attr_uav'].shape[-1])
        ea_adr = data['edge_attr_adr'].view(-1, data['edge_attr_adr'].shape[-1])
        ea_uav = self.ln_edge_uav(self.W_edge_uav(ea_uav))
        ea_adr = self.ln_edge_adr(self.W_edge_adr(ea_adr))

        edge_index = data['edge_index'].view(data['edge_index'].shape[1], -1)
        mask_uav   = data['mask_adjacency_uav']
        mask_adr   = data['mask_adjacency_adr']

        x_uav = x + self.mode_token_uav
        x_adr = x + self.mode_token_adr

        for conv_u, conv_a in zip(self.convs_uav, self.convs_adr):
            x_uav = conv_u(x_uav, edge_index, ea_uav, mask_uav, batch_size)
            x_adr = conv_a(x_adr, edge_index, ea_adr, mask_adr, batch_size)

        x_uav = self.ln_final_uav(x_uav + self.ff_uav(x_uav))
        x_adr = self.ln_final_adr(x_adr + self.ff_adr(x_adr))

        return (x_uav.reshape(batch_size, n_total, self.hidden_node_dim),
                x_adr.reshape(batch_size, n_total, self.hidden_node_dim))

class Encoder(nn.Module):
    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=4, hidden_edge_dim=16, conv_layers=3):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim

        self.W_depot = nn.Linear(input_node_dim, hidden_node_dim, bias=False)
        self.W_pickup = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        # Symmetric with W_pickup: delivery embedding also sees its paired pickup features
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        self.W_edge_uav = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.W_edge_adr = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)

        self.ln_node = nn.LayerNorm(hidden_node_dim)
        self.ln_edge_uav = nn.LayerNorm(hidden_edge_dim)
        self.ln_edge_adr = nn.LayerNorm(hidden_edge_dim)

        self.mode_token_uav = nn.Parameter(torch.zeros(hidden_node_dim))
        self.mode_token_adr = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.mode_token_uav, std=0.02)
        nn.init.normal_(self.mode_token_adr, std=0.02)

        self.convs_uav = nn.ModuleList([
            HetGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
            for _ in range(conv_layers)
        ])
        self.convs_adr = nn.ModuleList([
            HetGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
            for _ in range(conv_layers)
        ])

        self.ff_uav = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ff_adr = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ln_final_uav = nn.LayerNorm(hidden_node_dim)
        self.ln_final_adr = nn.LayerNorm(hidden_node_dim)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data['x'].shape[0]
        feat = data['x']
        n_total = feat.shape[1]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = n_total - num_depots
        n_pickup = num_nodes // 2

        depot_f = feat[:, :num_depots, :].reshape(batch_size * num_depots, -1)
        pickup_f = feat[:, num_depots:num_depots + n_pickup, :]
        delivery_f = feat[:, num_depots + n_pickup:, :]
        pickup_paired   = torch.cat([pickup_f,   delivery_f], dim=-1).reshape(batch_size * n_pickup, -1)
        delivery_paired = torch.cat([delivery_f, pickup_f],   dim=-1).reshape(batch_size * n_pickup, -1)

        emb_depot    = self.W_depot(depot_f)
        emb_pickup   = self.W_pickup(pickup_paired)
        emb_delivery = self.W_delivery(delivery_paired)

        x = torch.cat([
            emb_depot.view(batch_size, num_depots, -1),
            emb_pickup.view(batch_size, n_pickup, -1),
            emb_delivery.view(batch_size, n_pickup, -1),
        ], dim=1).view(-1, self.hidden_node_dim)
        x = self.ln_node(x)

        ea_uav = data['edge_attr_uav'].view(-1, data['edge_attr_uav'].shape[-1])
        ea_adr = data['edge_attr_adr'].view(-1, data['edge_attr_adr'].shape[-1])
        ea_uav = self.ln_edge_uav(self.W_edge_uav(ea_uav))
        ea_adr = self.ln_edge_adr(self.W_edge_adr(ea_adr))

        edge_index = data['edge_index'].view(data['edge_index'].shape[1], -1)
        mask_uav = data['mask_adjacency_uav']
        mask_adr = data['mask_adjacency_adr']

        x_uav = x + self.mode_token_uav
        x_adr = x + self.mode_token_adr

        for conv_u, conv_a in zip(self.convs_uav, self.convs_adr):
            x_uav = conv_u(x_uav, edge_index, ea_uav, num_depots, num_nodes,
                           mask_uav, batch_size)
            x_adr = conv_a(x_adr, edge_index, ea_adr, num_depots, num_nodes,
                           mask_adr, batch_size)

        x_uav = self.ln_final_uav(x_uav + self.ff_uav(x_uav))
        x_adr = self.ln_final_adr(x_adr + self.ff_adr(x_adr))

        x_uav = x_uav.reshape(batch_size, n_total, self.hidden_node_dim)
        x_adr = x_adr.reshape(batch_size, n_total, self.hidden_node_dim)

        return x_uav, x_adr

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.norm = 1.0 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)

    def precompute(self, context):
        B, N, _ = context.shape
        K = self.k(context).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(context).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        return K, V

    def forward(self, Q_raw, K, V, mask):
        B, A, _ = Q_raw.shape
        Q = self.w(Q_raw).view(B, A, self.n_heads, self.head_dim).transpose(1, 2)
        compat = self.norm * torch.matmul(Q, K.transpose(2, 3))
        mask = mask.unsqueeze(1).expand_as(compat)
        compat = compat.masked_fill(mask.bool(), float(-1e9))
        scores = F.softmax(compat, dim=-1)
        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(B, A, self.hidden_dim)
        return self.fc(out)

def _parallel_select(u: torch.Tensor, num_depots: int,
                     greedy: bool) -> tuple:
    B, A, N = u.shape
    dev = u.device
    valid = u > -1e8

    if not greedy:
        gumbel = -torch.log(
            -torch.log(torch.rand_like(u).clamp(min=1e-10)) + 1e-10
        )
        u_g = u + gumbel
    else:
        u_g = u

    winner = u_g.argmax(dim=1)

    ag_ids = torch.arange(A, device=dev).view(1, A, 1)
    agent_wins = (winner.unsqueeze(1) == ag_ids) & valid
    agent_wins[:, :, :num_depots] = valid[:, :, :num_depots]

    u_won = u_g.masked_fill(~agent_wins, float('-inf'))
    no_win = (u_won == float('-inf')).all(dim=-1)

    u_fallback = u_g.masked_fill(~valid, float('-inf'))
    u_won = torch.where(no_win.unsqueeze(-1), u_fallback, u_won)

    all_masked = (u_won == float('-inf')).all(dim=-1)
    u_won = u_won.masked_fill(all_masked.unsqueeze(-1), 0.0)

    selected = u_won.argmax(dim=-1)

    lp = F.log_softmax(
        u.masked_fill(~valid, float('-inf')).clamp(min=-1e9), dim=-1
    )
    lp = lp.masked_fill(all_masked.unsqueeze(-1), 0.0)
    log_p = lp.gather(2, selected.unsqueeze(2)).squeeze(2)

    return selected, log_p

class CoopDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.mha_uav = MultiHeadAttention(n_heads, 1, input_dim, hidden_dim)
        self.mha_adr = MultiHeadAttention(n_heads, 1, input_dim, hidden_dim)
        self.k_proj_uav = nn.Linear(input_dim, hidden_dim, bias=False)
        self.k_proj_adr = nn.Linear(input_dim, hidden_dim, bias=False)

        self.agent_id_emb_uav = nn.Parameter(torch.zeros(16, hidden_dim))
        self.agent_id_emb_adr = nn.Parameter(torch.zeros(16, hidden_dim))
        nn.init.normal_(self.agent_id_emb_uav, std=0.02)
        nn.init.normal_(self.agent_id_emb_adr, std=0.02)

        self.Wd_uav     = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.Wd_adr     = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.dyn_proj_uav = nn.Linear(3, hidden_dim, bias=True)
        self.dyn_proj_adr = nn.Linear(3, hidden_dim, bias=True)
        self.ln_uav = nn.LayerNorm(hidden_dim)
        self.ln_adr = nn.LayerNorm(hidden_dim)

        self.dist_compat_uav = nn.Linear(1, 1, bias=True)
        self.dist_compat_adr = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.dist_compat_uav.weight, -1.0)
        nn.init.constant_(self.dist_compat_uav.bias,    0.0)
        nn.init.constant_(self.dist_compat_adr.weight, -1.0)
        nn.init.constant_(self.dist_compat_adr.bias,    0.0)
        self.h_empty_uav = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.h_empty_adr = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.fc1_uav = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1_adr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cross_attn_uav = nn.MultiheadAttention(hidden_dim, n_heads,
                                                     batch_first=True, bias=False, dropout=0.0)
        self.cross_attn_adr = nn.MultiheadAttention(hidden_dim, n_heads,
                                                     batch_first=True, bias=False, dropout=0.0)
        self.ln_cross_uav = nn.LayerNorm(hidden_dim)
        self.ln_cross_adr = nn.LayerNorm(hidden_dim)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
        nn.init.constant_(self.dist_compat_uav.weight, -1.0)
        nn.init.constant_(self.dist_compat_adr.weight, -1.0)

        self.urgency_proj_uav = nn.Linear(1, 1, bias=True)
        self.urgency_proj_adr = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.urgency_proj_uav.weight, 0.0)
        nn.init.constant_(self.urgency_proj_uav.bias,   0.0)
        nn.init.constant_(self.urgency_proj_adr.weight, 0.0)
        nn.init.constant_(self.urgency_proj_adr.bias,   0.0)

    def _compatibility(self, q_vec, K, mask, T=1.0):
        norm = 1.0 / math.sqrt(K.shape[-1])
        compat = norm * torch.bmm(q_vec, K.transpose(1, 2))
        compat = 10.0 * torch.tanh(compat / T)
        compat = compat.masked_fill(mask.bool(), float('-inf'))
        return compat

    def forward(self, emb_uav, emb_adr,
                capacity, demand, battery, time_window,
                num_depots, num_nodes, num_uav, num_adr,
                edge_attr_uav, edge_attr_adr, T=1.0, greedy=False,
                acc_uav=None, acc_adr=None, parallel_select=False,
                n_depots_uav=None, initial_visited=None):

        batch_size, _, H = emb_uav.shape
        n_agents = num_uav + num_adr

        mask1 = emb_uav.new_zeros((batch_size, n_agents, emb_uav.size(1)))
        if initial_visited is not None:
            mask1 = initial_visited.unsqueeze(1).expand(batch_size, n_agents, -1).clone()
        mask = emb_uav.new_zeros((batch_size, n_agents, emb_uav.size(1)))

        battery = battery.view(batch_size, n_agents).unsqueeze(2).float()
        capacity = capacity.view(batch_size, n_agents).unsqueeze(2).float()
        T_t = torch.zeros(batch_size, n_agents, 1, device=emb_uav.device)

        E_uav = battery[0, 0].item()
        E_adr = battery[0, num_uav].item() if num_adr > 0 else 1.0
        E = [E_uav, E_adr]
        demands = demand.view(batch_size, emb_uav.size(1))
        time_window = time_window.view(batch_size, emb_uav.size(1))
        edge_attr_uav = edge_attr_uav.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        edge_attr_adr = edge_attr_adr.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

        K_uav, V_uav = self.mha_uav.precompute(emb_uav)
        K_adr, V_adr = self.mha_adr.precompute(emb_adr)
        K_all_uav = self.k_proj_uav(emb_uav)
        K_all_adr = self.k_proj_adr(emb_adr)

        n_pickup = num_nodes // 2

        deadline = time_window.clone()
        deadline[:, num_depots:num_depots + n_pickup] = (
            time_window[:, num_depots + n_pickup:num_depots + num_nodes]
        )

        id_emb_uav = self.agent_id_emb_uav[:num_uav].unsqueeze(0).expand(batch_size, -1, -1)
        id_emb_adr = self.agent_id_emb_adr[:num_adr].unsqueeze(0).expand(batch_size, -1, -1)

        onboard_uav_sum = emb_uav.new_zeros(batch_size, num_uav, H)
        onboard_uav_cnt = emb_uav.new_zeros(batch_size, num_uav, 1)
        onboard_adr_sum = emb_adr.new_zeros(batch_size, num_adr, H)
        onboard_adr_cnt = emb_adr.new_zeros(batch_size, num_adr, 1)

        capacity_init = capacity.clone().clamp(min=1.0)
        E_tensor = battery.new_tensor(
            [max(E_uav, 1.0)] * num_uav + [max(E_adr, 1.0)] * num_adr
        ).view(1, n_agents, 1)

        log_ps, actions, time_log = [], [], []
        i = 0
        _max_steps = num_nodes + n_agents * 4

        # Pre-compute loop-invariant tensors (avoid repeated allocation inside while loop)
        _batch_range_uav = torch.arange(batch_size, device=emb_uav.device).unsqueeze(1).expand(-1, num_uav)
        _batch_range_adr = torch.arange(batch_size, device=emb_uav.device).unsqueeze(1).expand(-1, num_adr)
        _deadline_exp    = deadline.unsqueeze(1)   # [B, 1, N] — deadline doesn't change
        # Sequential selection: pre-allocate; reset in-place each step
        _sel_buf  = emb_uav.new_zeros(batch_size, n_agents, dtype=torch.long)
        _ask_buf  = emb_uav.new_zeros(batch_size, emb_uav.size(1), dtype=torch.bool)
        # Agent permutation — greedy order is fixed; stochastic is re-drawn each step
        _perm_greedy = torch.arange(n_agents, device=emb_uav.device)

        while (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any() and i < _max_steps:
            if i == 0:
                if n_depots_uav is not None and n_depots_uav > 0:
                    n_depots_adr_val = num_depots - n_depots_uav
                    uav_depot = torch.remainder(
                        torch.arange(num_uav, device=emb_uav.device), n_depots_uav
                    )
                    adr_depot = n_depots_uav + torch.remainder(
                        torch.arange(num_adr, device=emb_uav.device),
                        max(n_depots_adr_val, 1),
                    )
                    depot_idx = torch.cat([uav_depot, adr_depot]).unsqueeze(0).expand(batch_size, -1)
                else:
                    depot_idx = torch.remainder(
                        torch.arange(n_agents, device=emb_uav.device), num_depots
                    ).unsqueeze(0).expand(batch_size, -1)
                index = depot_idx
                s_t_uav = torch.gather(emb_uav, 1,
                    depot_idx[:, :num_uav].unsqueeze(2).expand(-1, -1, H))
                s_t_adr = torch.gather(emb_adr, 1,
                    depot_idx[:, num_uav:].unsqueeze(2).expand(-1, -1, H))
                s_t = torch.cat([s_t_uav, s_t_adr], dim=1)
                actions.append(index.unsqueeze(2))

            remaining = (mask1[:, :, num_depots:].max(dim=1)[0] == 0).float()
            r_sum = remaining.sum(1, keepdim=True).clamp(min=1.0)
            pool_uav = (emb_uav[:, num_depots:, :] * remaining.unsqueeze(-1)).sum(1) / r_sum
            pool_adr = (emb_adr[:, num_depots:, :] * remaining.unsqueeze(-1)).sum(1) / r_sum

            dyn = torch.cat([
                capacity / capacity_init,
                T_t / 120.0,
                battery / E_tensor,
            ], dim=-1)

            onboard_uav = torch.where(
                onboard_uav_cnt > 0,
                onboard_uav_sum / onboard_uav_cnt.clamp(min=1.0),
                self.h_empty_uav.expand(batch_size, num_uav, H),
            )
            onboard_adr = torch.where(
                onboard_adr_cnt > 0,
                onboard_adr_sum / onboard_adr_cnt.clamp(min=1.0),
                self.h_empty_adr.expand(batch_size, num_adr, H),
            )

            ctx_uav_emb = self.Wd_uav(
                torch.cat([s_t[:, :num_uav, :] + id_emb_uav, onboard_uav], dim=-1).view(-1, H * 2)
            ).view(batch_size, num_uav, H)
            ctx_adr_emb = self.Wd_adr(
                torch.cat([s_t[:, num_uav:, :] + id_emb_adr, onboard_adr], dim=-1).view(-1, H * 2)
            ).view(batch_size, num_adr, H)
            ctx_uav = self.ln_uav(ctx_uav_emb + self.dyn_proj_uav(dyn[:, :num_uav, :].reshape(-1, 3)).view(batch_size, num_uav, H))
            ctx_adr = self.ln_adr(ctx_adr_emb + self.dyn_proj_adr(dyn[:, num_uav:, :].reshape(-1, 3)).view(batch_size, num_adr, H))

            ctx_uav_pre, ctx_adr_pre = ctx_uav, ctx_adr
            cross_uav, _ = self.cross_attn_uav(ctx_uav_pre, ctx_adr_pre, ctx_adr_pre)
            cross_adr, _ = self.cross_attn_adr(ctx_adr_pre, ctx_uav_pre, ctx_uav_pre)
            ctx_uav = self.ln_cross_uav(ctx_uav + cross_uav)
            ctx_adr = self.ln_cross_adr(ctx_adr + cross_adr)

            dec_uav = (pool_uav.unsqueeze(1).expand(-1, num_uav, -1)
                       + self.fc1_uav(ctx_uav))
            dec_adr = (pool_adr.unsqueeze(1).expand(-1, num_adr, -1)
                       + self.fc1_adr(ctx_adr))
            dec_input = torch.cat([dec_uav, dec_adr], dim=1)

            if i == 0:
                mask, mask1 = update_mask(demands, capacity, index, mask1, battery, num_uav, E, i,
                                          acc_uav, acc_adr,
                                          edge_attr_d=edge_attr_uav, edge_attr_r=edge_attr_adr,
                                          num_depots=num_depots)

            q_uav = self.mha_uav(dec_input[:, :num_uav, :], K_uav, V_uav, mask[:, :num_uav, :])
            q_adr = self.mha_adr(dec_input[:, num_uav:, :], K_adr, V_adr, mask[:, num_uav:, :])

            dist_uav = edge_attr_uav[_batch_range_uav, index[:, :num_uav]]
            dist_adr = edge_attr_adr[_batch_range_adr, index[:, num_uav:]]
            dist_bias_uav = self.dist_compat_uav(dist_uav.unsqueeze(-1)).squeeze(-1)
            dist_bias_adr = self.dist_compat_adr(dist_adr.unsqueeze(-1)).squeeze(-1)

            slack = (_deadline_exp - T_t) / 120.0
            urgency_bias_uav = self.urgency_proj_uav(
                slack[:, :num_uav, :].unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, :num_uav, :].bool(), 0.0)
            urgency_bias_adr = self.urgency_proj_adr(
                slack[:, num_uav:, :].unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, num_uav:, :].bool(), 0.0)

            u_uav = (self._compatibility(q_uav, K_all_uav, mask[:, :num_uav, :], T)
                     + dist_bias_uav.masked_fill(mask[:, :num_uav, :].bool(), 0.0)
                     + urgency_bias_uav)
            u_adr = (self._compatibility(q_adr, K_all_adr, mask[:, num_uav:, :], T)
                     + dist_bias_adr.masked_fill(mask[:, num_uav:, :].bool(), 0.0)
                     + urgency_bias_adr)
            u = torch.cat([u_uav, u_adr], dim=1)

            if parallel_select:
                index, log_p = _parallel_select(u, num_depots, greedy)
            else:
                # Reuse pre-allocated buffers (avoid per-step malloc)
                _sel_buf.zero_()
                _ask_buf.zero_()
                log_p_list = [None] * n_agents
                perm = (_perm_greedy if greedy
                        else torch.randperm(n_agents, device=emb_uav.device))
                for step_i in range(n_agents):
                    ag = int(perm[step_i].item())
                    mu = u[:, ag, :].masked_fill(_ask_buf, float('-inf'))
                    all_masked_ag = (mu == float('-inf')).all(dim=-1, keepdim=True)
                    mu_safe = mu.masked_fill(all_masked_ag, 0.0)
                    log_probs = F.log_softmax(mu_safe.clamp(min=-1e9), dim=-1)
                    if not greedy:
                        mi = torch.multinomial(log_probs.detach().exp(), 1).squeeze(1)
                    else:
                        mi = mu_safe.max(dim=-1)[1]
                    _sel_buf[:, ag] = mi
                    lp = log_probs.gather(1, mi.unsqueeze(1)).squeeze(1)
                    lp = lp.masked_fill(all_masked_ag.squeeze(-1), 0.0)
                    log_p_list[ag] = lp
                    _ask_buf.scatter_(1, mi.unsqueeze(1), True)
                    # Depot chosen: that depot stays available to other agents
                    depot_mask = mi.lt(num_depots)
                    if depot_mask.any():
                        _ask_buf[depot_mask, :num_depots] = False

                log_p = torch.stack(log_p_list, dim=1)
                index = _sel_buf
            actions.append(index.unsqueeze(2))

            is_done = (mask1[:, :, num_depots:].max(dim=1)[0].sum(1).unsqueeze(1)
                       .expand(batch_size, n_agents) >= (emb_uav.size(1) - num_depots)).float()
            log_p = log_p * (1.0 - is_done)
            log_ps.append(log_p.unsqueeze(2))

            capacity, T_t, battery = update_state(
                demands, time_window, battery, T_t, capacity, E,
                num_uav, actions, edge_attr_uav, edge_attr_adr,
                num_depots=num_depots,
            )
            capacity = capacity.unsqueeze(2)
            T_t = T_t.unsqueeze(2)
            battery = battery.unsqueeze(2)
            time_log.append(T_t)

            mask, mask1 = update_mask(demands, capacity, index, mask1, battery, num_uav, E, i,
                                      acc_uav, acc_adr,
                                      edge_attr_d=edge_attr_uav, edge_attr_r=edge_attr_adr,
                                      num_depots=num_depots)

            s_t_uav = torch.gather(emb_uav, 1,
                index[:, :num_uav].unsqueeze(2).expand(-1, -1, H))
            s_t_adr = torch.gather(emb_adr, 1,
                index[:, num_uav:].unsqueeze(2).expand(-1, -1, H))
            s_t = torch.cat([s_t_uav, s_t_adr], dim=1)

            idx_uav = index[:, :num_uav]
            is_pickup_uav = (idx_uav >= num_depots) & (idx_uav < num_depots + n_pickup)
            is_delivery_uav = idx_uav >= (num_depots + n_pickup)

            delivery_idx_uav = (idx_uav + n_pickup).clamp(max=emb_uav.size(1) - 1)
            paired_del_emb_uav = torch.gather(
                emb_uav, 1, delivery_idx_uav.unsqueeze(-1).expand(-1, -1, H)
            )
            cur_emb_uav = torch.gather(emb_uav, 1, idx_uav.unsqueeze(-1).expand(-1, -1, H))

            p_u = is_pickup_uav.unsqueeze(-1)
            d_u = is_delivery_uav.unsqueeze(-1)
            onboard_uav_sum = (onboard_uav_sum
                               + paired_del_emb_uav * p_u.float()
                               - cur_emb_uav * d_u.float())
            onboard_uav_cnt = (onboard_uav_cnt
                               + p_u.float()
                               - d_u.float()).clamp(min=0.0)

            idx_adr = index[:, num_uav:]
            is_pickup_adr = (idx_adr >= num_depots) & (idx_adr < num_depots + n_pickup)
            is_delivery_adr = idx_adr >= (num_depots + n_pickup)

            delivery_idx_adr = (idx_adr + n_pickup).clamp(max=emb_adr.size(1) - 1)
            paired_del_emb_adr = torch.gather(
                emb_adr, 1, delivery_idx_adr.unsqueeze(-1).expand(-1, -1, H)
            )
            cur_emb_adr = torch.gather(emb_adr, 1, idx_adr.unsqueeze(-1).expand(-1, -1, H))

            p_a = is_pickup_adr.unsqueeze(-1)
            d_a = is_delivery_adr.unsqueeze(-1)
            onboard_adr_sum = (onboard_adr_sum
                               + paired_del_emb_adr * p_a.float()
                               - cur_emb_adr * d_a.float())
            onboard_adr_cnt = (onboard_adr_cnt
                               + p_a.float()
                               - d_a.float()).clamp(min=0.0)

            i += 1

        actions = torch.cat(actions, dim=2)
        Time = torch.cat(time_log, dim=2)
        return actions, torch.cat(log_ps, dim=2).sum(dim=2), Time

class Model(nn.Module):
    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=4, hidden_edge_dim=16, conv_layers=3,
                 arch='hetgat'):
        super().__init__()
        if arch == 'simplegat':
            self.encoder = SimpleGatEncoder(input_node_dim, hidden_node_dim,
                                            input_edge_dim, hidden_edge_dim, conv_layers)
        else:
            self.encoder = Encoder(input_node_dim, hidden_node_dim,
                                   input_edge_dim, hidden_edge_dim, conv_layers)
        self.decoder = CoopDecoder(hidden_node_dim, hidden_node_dim)

    def forward(self, data, num_uav, num_adr, greedy=False, T=1.0,
                checkpoint_encoder=False, training=False, parallel_select=False,
                initial_visited=None):
        import torch.utils.checkpoint as cp
        if checkpoint_encoder and training:
            x_uav, x_adr = cp.checkpoint(self.encoder, data, use_reentrant=False)
        else:
            x_uav, x_adr = self.encoder(data)

        batch_size = data['x'].shape[0]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = x_uav.shape[1] - num_depots

        nd_uav = data.get('n_depots_uav', None)
        n_depots_uav = (int(nd_uav[0]) if hasattr(nd_uav, '__len__') else int(nd_uav)) if nd_uav is not None else None

        ea_uav = data['edge_attr_d']
        ea_adr = data['edge_attr_r']

        acc_uav = data['x'][:, :, 8]
        acc_adr = data['x'][:, :, 9]

        actions, log_p, time = self.decoder(
            x_uav, x_adr,
            data['capacity'], data['demand'], data['battery'],
            data['time_window'], num_depots, num_nodes,
            num_uav, num_adr, ea_uav, ea_adr, T, greedy,
            acc_uav=acc_uav, acc_adr=acc_adr,
            parallel_select=parallel_select,
            n_depots_uav=n_depots_uav,
            initial_visited=initial_visited,
        )
        return actions, log_p, time

from vrpUpdate import update_mask, update_state