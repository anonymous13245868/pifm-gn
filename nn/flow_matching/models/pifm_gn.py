import torch
from torch import nn
import torch.nn.functional as F
import os

from ..flow_matching_model import FlowMatchingModel
from ...blocks import SinusoidalTimeEmbedding
from ...models.multi_scale_gnn import MultiScaleGnn
from ....graph import Graph
from torch_scatter import scatter_add

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

# def graph_curl_operator(potential, edge_index, pos):
#     grad = compute_graph_gradient(potential, edge_index, pos)
    
#     curl = torch.zeros(pos.size(0), 3, device=pos.device)
    
#     dim = pos.size(1)
#     pot_dim = potential.size(1)
    
#     if dim == 3 and pot_dim == 3:
#         # 3D情况
#         curl[:, 0] = grad[:, 1, 2] - grad[:, 2, 1]
#         curl[:, 1] = grad[:, 2, 0] - grad[:, 0, 2]
#         curl[:, 2] = grad[:, 0, 1] - grad[:, 1, 0]
#     elif dim == 2:
#         if pot_dim == 1:
#             curl[:, 0] = -grad[:, 1, 0]
#             curl[:, 1] = grad[:, 0, 0]
#         else:
#             curl[:, 2] = grad[:, 0, 1] - grad[:, 1, 0]
    
#     return curl

def compute_graph_divergence(vector_field, edge_index, pos, eps=1e-6):
    grads = compute_graph_gradient(vector_field, edge_index, pos, eps)
    return grads 


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
    """简化版径向基函数距离嵌入"""
    def __init__(self, num_rbf=16, cutoff=10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.widths = nn.Parameter(torch.tensor(cutoff / num_rbf), requires_grad=False)
        
    def forward(self, distances):
        dist_feat = torch.exp(-((distances - self.centers)**2) / (2 * self.widths**2))
        return dist_feat

class ConservationFlowMatchingGraphNet_MIX(FlowMatchingModel):

    def __init__(self, *args, **kwargs):
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
        
        if 'in_edge_features' in arch:  
             self.cond_edge_features = arch['in_edge_features'] + self.cond_edge_features
        
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
        
        self.rbf_embedding = RBFDistanceEmbedding(num_rbf=self.rbf_dim)
        
        self.r_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.fnns_width),
            nn.Linear(self.fnns_width, self.emb_width),
            nn.SELU(),
        )
        
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features + self.cond_node_features,
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
            dim               = self.dim,
            scalar_rel_pos    = self.scalar_rel_pos,
        )
        
        # 1) direct branch
        self.direct_decoder = nn.Linear(self.fnns_width,
                                        self.out_node_features)
        # 2) potential→curl branch
        self.potential_network = nn.Sequential(
            nn.Linear(self.fnns_width, self.fnns_width),
            nn.SELU(),
            nn.Linear(self.fnns_width, self.potential_dim),
        )

        self.curl_weight = arch.get('curl_weight', 0.2)
        self.div_penalty = arch.get('div_penalty', 0.1)

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
        assert hasattr(graph, 'r'), 
        assert hasattr(graph, 'field_r'),
        assert hasattr(graph, 'pos'), 
        assert hasattr(graph, 'edge_index'),
        
        r = graph.r
        field_r = graph.field_r
        pos = graph.pos
        edge_index = graph.edge_index
        batch = graph.batch
        emb = self.r_embedding(r) 
        
        node_features = [field_r]
        for attr in ['cond', 'field', 'loc', 'glob', 'omega']:
            feat = graph.get(attr)
            if feat is not None:
                # print(attr,feat.shape)
                node_features.append(feat)
        v = self.node_encoder(torch.cat(node_features, dim=1))
        
        emb_proj = self.r_encoder[0](emb)
        v = torch.cat([v, emb_proj[batch]], dim=1)
        v = self.r_encoder[2](self.r_encoder[1](v)) 
            
        src, dst = graph.edge_index
        edge_vec = graph.pos[dst] - graph.pos[src]
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        
        edge_rbf = self.rbf_embedding(edge_length)
        
        edge_features = [edge_rbf]
        if hasattr(graph, 'edge_attr'):
            edge_features.insert(0, graph.edge_attr)  
        if graph.get('edge_cond') is not None:
            edge_features.append(graph.get('edge_cond')) 
            
        e = self.edge_encoder(torch.cat(edge_features, dim=1))
        
        v, _ = self.propagator(graph, v, e, emb)  # v: (N, fnns_width)

        # direct decoder
        v_direct = self.direct_decoder(v)         # (N, out_node_features)

        # potential -> curl decoder
        potential = self.potential_network(v)     # (N, potential_dim)
        v_curl = graph_curl_operator(
            potential, graph.edge_index, graph.pos)  # (N, self.dim)
        v_curl = self.proj_curl(v_curl)           # (N, out_node_features)

        vector_field = (1 - self.curl_weight) * v_direct + self.curl_weight * v_curl
        return vector_field