import torch
from torch import nn
import torch.nn.functional as F
import os

from ..flow_matching_model import FlowMatchingModel
from ...blocks import SinusoidalTimeEmbedding
from ...models.multi_scale_gnn import MultiScaleGnn
from ...models.vgae import VGAE
from ....graph import Graph

def compute_graph_gradient(node_values, edge_index, pos, eps=1e-6):
    src, dst = edge_index
    
    edge_vec = pos[dst] - pos[src]
    edge_length_squared = torch.sum(edge_vec**2, dim=1, keepdim=True) + eps
    edge_length = torch.sqrt(edge_length_squared)
    edge_dir = edge_vec / edge_length
    
    value_diff = node_values[dst] - node_values[src]
    
    directional_deriv = (value_diff / edge_length).unsqueeze(1)
    contribution = directional_deriv * edge_dir.unsqueeze(2)
    
    ones = torch.ones(dst.size(0), device=dst.device)
    degrees = torch.zeros(pos.size(0), device=pos.device).scatter_add_(0, dst, ones) + 1e-6
    
    gradients = torch.zeros(pos.size(0), pos.size(1), node_values.size(1), device=pos.device)
    gradients.index_add_(0, dst, contribution)
    
    return gradients / degrees.view(-1, 1, 1)

def graph_curl_operator(potential, edge_index, pos, eps=1e-6):
    grad = compute_graph_gradient(potential, edge_index, pos, eps)

    N, D = pos.size()
    pot_dim = potential.size(1)

    if D == 3 and pot_dim == 3:
        curl = torch.stack([
            grad[:,1,2] - grad[:,2,1],
            grad[:,2,0] - grad[:,0,2],
            grad[:,0,1] - grad[:,1,0],
        ], dim=1)  # (N,3)
    elif D == 2 and pot_dim == 1:
        gx = grad[:,0,0]   # ∂ψ/∂x
        gy = grad[:,1,0]   # ∂ψ/∂y
        curl = torch.stack([gy, -gx], dim=1)  # (N,2)
    else:
        raise ValueError(f"Unsupported dim={D}, potential_dim={pot_dim}")
    return curl

class RBFDistanceEmbedding(nn.Module):
    def __init__(self, num_rbf=16, cutoff=10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.widths = nn.Parameter(torch.tensor(cutoff / num_rbf), requires_grad=False)
        
    def forward(self, distances):
        dist_feat = torch.exp(-((distances - self.centers)**2) / (2 * self.widths**2))
        return dist_feat

class LatentConservationFlowMatchingGraphNet(FlowMatchingModel):

    def __init__(
        self,
        autoencoder_checkpoint: str,
        *args,
        **kwargs
    ) -> None: 
        self.autoencoder_checkpoint = autoencoder_checkpoint
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        self.in_node_features   = arch['in_node_features']
        self.cond_node_features = arch.get('cond_node_features', 0)
        self.cond_edge_features = arch.get('cond_edge_features', 0)
        self.depths             = arch['depths']
        self.fnns_depth         = arch.get('fnns_depth', 2)
        self.fnns_width         = arch['fnns_width']
        self.aggr               = arch.get('aggr', 'mean')
        self.dropout            = arch.get('dropout', 0.0)
        self.emb_width          = arch.get('emb_width', self.fnns_width * 4)
        self.dim                = arch.get('dim', 2)
        self.scalar_rel_pos     = arch.get('scalar_rel_pos', True)
        
        self.potential_dim      = arch.get('potential_dim', 3)
        if self.dim == 2:
            self.potential_dim = 1
        self.rbf_dim            = arch.get('rbf_dim', 16)
        self.curl_weight        = arch.get('curl_weight', 0.2)
        self.div_penalty        = arch.get('div_penalty', 0.1)
        
        assert self.in_node_features > 0, 
        assert self.cond_node_features >= 0, 
        assert len(self.depths) > 0, 
        assert isinstance(self.depths, list), 
        assert all([isinstance(depth, int) for depth in self.depths]), 
        assert all([depth > 0 for depth in self.depths]),
        assert self.fnns_depth >= 2, 
        assert self.fnns_width > 0, 
        assert self.aggr in ('mean', 'sum'), 
        assert self.dropout >= 0.0 and self.dropout < 1.0, 
        
        self.out_node_features = self.in_node_features
        
        self.autoencoder = VGAE(
            checkpoint = self.autoencoder_checkpoint,
            device     = self.device,
        )
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.scale_0 = len(self.autoencoder.arch['depths']) - 1
        
        self.rbf_embedding = RBFDistanceEmbedding(num_rbf=self.rbf_dim)
        
        self.r_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.fnns_width),
            nn.Linear(self.fnns_width, self.emb_width),
            nn.SELU(),
        )
        
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features,
            out_features = self.fnns_width,
        )
        
        self.cond_encoder = nn.Linear(
            in_features  = self.cond_node_features,
            out_features = self.fnns_width,
        )
        
        self.r_encoder = nn.ModuleList([
            nn.Linear(self.emb_width, self.fnns_width),     
            nn.SELU(),                                    
            nn.Linear(self.fnns_width * 2, self.fnns_width),
        ])
        
        self.edge_encoder = nn.Linear(
            in_features  = self.cond_edge_features + self.rbf_dim,
            out_features = self.fnns_width,
        )
        
        self.propagator = MultiScaleGnn(
            depths            = self.depths,
            fnns_depth        = self.fnns_depth,
            fnns_width        = self.fnns_width,
            emb_features      = self.emb_width,
            aggr              = self.aggr,
            activation        = nn.SELU,
            dropout           = self.dropout,
            scale_0           = self.scale_0,
            dim               = self.dim,
            scalar_rel_pos    = self.scalar_rel_pos,
        )
        
        self.direct_decoder = nn.Linear(
            in_features  = self.fnns_width,
            out_features = self.out_node_features,
        )
        
        self.potential_network = nn.Sequential(
            nn.Linear(self.fnns_width, self.fnns_width),
            nn.SELU(),
            nn.Linear(self.fnns_width, self.potential_dim),
        )
        
        if self.dim != self.out_node_features:
            self.proj_curl = nn.Linear(self.dim, self.out_node_features)
        else:
            self.proj_curl = nn.Identity()
 
    @property
    def num_fields(self) -> int:
        return self.out_node_features
    
    def reset_parameters(self):
        modules = [module for module in self.children() if hasattr(module, 'reset_parameters')]
        for module in modules:
            module.reset_parameters()

    def forward(self, graph: Graph) -> torch.Tensor:
        # 验证必要的图属性
        assert hasattr(graph, 'c_latent'), 
        assert hasattr(graph, 'e_latent'), 
        assert hasattr(graph, 'r'), 
        assert hasattr(graph, 'field_r'), 
        assert hasattr(graph, 'pos'), 
        assert hasattr(graph, 'edge_index'), 
        
        emb = self.r_embedding(graph.r)  
        i = self.scale_0 + 1   
        x_latent = self.node_encoder(graph.field_r) + self.cond_encoder(graph.c_latent)
        
        emb_proj = self.r_encoder[0](emb)
        x_latent = torch.cat([x_latent, emb_proj[graph.batch]], dim=1)
        for layer in self.r_encoder[1:]:
            x_latent = layer(x_latent)
        
        src, dst = graph.edge_index
        edge_vec = graph.pos[dst] - graph.pos[src]
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        
        edge_rbf = self.rbf_embedding(edge_length)
        
        e_latent = self.edge_encoder(torch.cat([graph.e_latent, edge_rbf], dim=1))
        
        x_latent, _ = self.propagator(graph, x_latent, e_latent, emb)
        
        # direct branch
        v_direct = self.direct_decoder(x_latent)
        
        # potential -> curl
        potential = self.potential_network(x_latent)
        pos_coarse        = getattr(graph, f'pos_{i}')
        edge_index_coarse = getattr(graph, f'edge_index_{i}')
        v_curl = graph_curl_operator(potential, edge_index_coarse, pos_coarse)
        v_curl = self.proj_curl(v_curl)
        
        vector_field = (1 - self.curl_weight) * v_direct + self.curl_weight * v_curl
        
        return vector_field