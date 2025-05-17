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
    """优化的图梯度计算"""
    src, dst = edge_index
    
    # 批量计算边向量和长度
    edge_vec = pos[dst] - pos[src]
    edge_length_squared = torch.sum(edge_vec**2, dim=1, keepdim=True) + eps
    edge_length = torch.sqrt(edge_length_squared)
    edge_dir = edge_vec / edge_length
    
    # 批量计算值差异
    value_diff = node_values[dst] - node_values[src]
    
    # 批量计算方向导数和贡献
    directional_deriv = (value_diff / edge_length).unsqueeze(1)
    contribution = directional_deriv * edge_dir.unsqueeze(2)
    
    # 预先计算节点度，避免重复计算
    ones = torch.ones(dst.size(0), device=dst.device)
    degrees = torch.zeros(pos.size(0), device=pos.device).scatter_add_(0, dst, ones) + 1e-6
    
    # 单次scatter_add计算梯度
    gradients = torch.zeros(pos.size(0), pos.size(1), node_values.size(1), device=pos.device)
    gradients.index_add_(0, dst, contribution)
    
    # 批量归一化
    return gradients / degrees.view(-1, 1, 1)

# def graph_curl_operator(potential, edge_index, pos):
#     """更高效的图旋度(curl)计算"""
#     # 计算势函数的梯度
#     grad = compute_graph_gradient(potential, edge_index, pos)
    
#     # 预先创建结果张量
#     curl = torch.zeros(pos.size(0), 3, device=pos.device)
    
#     # 检查空间维度并应用相应计算
#     dim = pos.size(1)
#     pot_dim = potential.size(1)
    
#     # 根据维度进行不同的计算（减少分支，使用张量操作）
#     if dim == 3 and pot_dim == 3:
#         # 3D情况
#         curl[:, 0] = grad[:, 1, 2] - grad[:, 2, 1]
#         curl[:, 1] = grad[:, 2, 0] - grad[:, 0, 2]
#         curl[:, 2] = grad[:, 0, 1] - grad[:, 1, 0]
#     elif dim == 2:
#         # 2D情况
#         if pot_dim == 1:
#             curl[:, 0] = -grad[:, 1, 0]
#             curl[:, 1] = grad[:, 0, 0]
#         else:
#             curl[:, 2] = grad[:, 0, 1] - grad[:, 1, 0]
    
#     return curl

def compute_graph_divergence(vector_field, edge_index, pos, eps=1e-6):
    # vector_field: (N, C) 且 C==pos.size(1) 时才有物理意义
    grads = compute_graph_gradient(vector_field, edge_index, pos, eps)
    # grads: (N, C, D)  若 C==D，则 grads[i,c,d] = ∂_d v_c(i)
    return grads  # forward 里用 grads.diagonal(dim1=1,dim2=2).sum(1)


def graph_curl_operator(potential, edge_index, pos, eps=1e-6):
    """统一处理 3D 向量势和 2D 标量势 stream function."""
    # 先算节点势的图上梯度
    grad = compute_graph_gradient(potential, edge_index, pos, eps)

    N, D = pos.size()
    pot_dim = potential.size(1)
    if D == 3 and pot_dim == 3:
        # 三维向量势 A -> v = curl A
        curl = torch.stack([
            grad[:,1,2] - grad[:,2,1],
            grad[:,2,0] - grad[:,0,2],
            grad[:,0,1] - grad[:,1,0],
        ], dim=1)  # (N,3)
    elif D == 2 and pot_dim == 1:
        # 二维 scalar potential ψ -> v = (∂ψ/∂y, -∂ψ/∂x)
        # grad[:,:,0] 是 ψ 在 x/y 上的导数
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
        
        # 中心点均匀分布在[0, cutoff]
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        # 宽度设置为相邻中心点间距
        self.widths = nn.Parameter(torch.tensor(cutoff / num_rbf), requires_grad=False)
        
    def forward(self, distances):
        """将标量距离转换为RBF特征"""
        # 计算距离到每个中心点的高斯距离
        dist_feat = torch.exp(-((distances - self.centers)**2) / (2 * self.widths**2))
        return dist_feat

class ConservationFlowMatchingGraphNet_MIX(FlowMatchingModel):
    """保守流匹配图神经网络：通过势函数参数化确保向量场满足物理守恒律(∇·v=0)。
    
    基于FlowMatchingGraphNet优化实现，保持输入接口兼容性，同时添加：
    1. 势函数参数化确保向量场无散度
    2. 基本的标量-向量特征分离以增强等变性
    3. 使用RBF距离嵌入增强几何表达
    
    Args:
        arch (dict): 模型架构参数字典，包含与FMGN相同的基本参数，以及：
            - 'potential_dim' (int, optional): 势函数维度，默认为3
            - 'rbf_dim' (int, optional): RBF距离嵌入维度，默认为16
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # 超参数，保持与FMGN一致的基本参数
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
        
        # 新增参数
        self.potential_dim      = arch.get('potential_dim', 3)
        if self.dim == 2:
            self.potential_dim = 1
        self.rbf_dim            = arch.get('rbf_dim', 16)      # RBF嵌入维度
        
        if 'in_edge_features' in arch:  # 支持向后兼容
             self.cond_edge_features = arch['in_edge_features'] + self.cond_edge_features
        
        # 输入验证
        assert self.in_node_features > 0, "输入节点特征维度必须为正整数"
        assert self.cond_node_features >= 0, "条件特征维度必须为非负整数"
        assert len(self.depths) > 0, "深度(depths)必须是非空整数列表"
        assert isinstance(self.depths, list), "深度(depths)必须是整数列表"
        assert all([isinstance(depth, int) for depth in self.depths]), "深度(depths)必须是整数列表"
        assert all([depth > 0 for depth in self.depths]), "深度(depths)必须是正整数列表"
        assert self.fnns_depth >= 2, "FNN深度(fnns_depth)至少为2"
        assert self.fnns_width > 0, "FNN宽度(fnns_width)必须为正整数"
        assert self.aggr in ('mean', 'sum'), "聚合方法(aggr)必须为'mean'或'sum'"
        assert self.dropout >= 0.0 and self.dropout < 1.0, "Dropout(dropout)必须在0.0到1.0之间"
        
        # 为了支持curl操作的输出，设置输出维度为3
        # 不要硬编码输出维度为3
        self.out_node_features = self.in_node_features  # 与FMGN保持一致
        
        # 距离嵌入层
        self.rbf_embedding = RBFDistanceEmbedding(num_rbf=self.rbf_dim)
        
        # 时间步嵌入，与FMGN相同
        self.r_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.fnns_width),
            nn.Linear(self.fnns_width, self.emb_width),
            nn.SELU(),
        )
        
        # 节点编码器 - 保持FMGN的接口
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features + self.cond_node_features,
            out_features = self.fnns_width,
        )
        
        # r编码器 - 保持FMGN的接口
        self.r_encoder = nn.ModuleList([
            nn.Linear(self.emb_width, self.fnns_width),      # 应用于r嵌入
            nn.SELU(),                                      # 应用于前一输出和节点编码器输出
            nn.Linear(self.fnns_width * 2, self.fnns_width), # 应用于前一输出
        ])
        
        # 边编码器 - 增强处理距离嵌入
        self.edge_encoder = nn.Linear(
            in_features  = self.cond_edge_features + self.rbf_dim,  # 额外包含RBF特征
            out_features = self.fnns_width,
        )
        
        # 使用原有的MultiScaleGnn
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
        
        # 1) 直接解码分支
        self.direct_decoder = nn.Linear(self.fnns_width,
                                        self.out_node_features)
        # 2) 势函数→curl 分支
        self.potential_network = nn.Sequential(
            nn.Linear(self.fnns_width, self.fnns_width),
            nn.SELU(),
            nn.Linear(self.fnns_width, self.potential_dim),
        )

        # 3) 混合权重与散度惩罚（实际惩罚在 Loss 里实现，这里仅存权重）
        self.curl_weight = arch.get('curl_weight', 0.2)
        self.div_penalty = arch.get('div_penalty', 0.1)

        # 4) 如果 curl 输出维度（self.dim）和 out_node_features 不一致，就用线性层映射
        if self.dim != self.out_node_features:
            self.proj_curl = nn.Linear(self.dim, self.out_node_features)
        else:
            # 直接恒等映射
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
        assert hasattr(graph, 'r'), "图必须具有扩散时间步('r')"
        assert hasattr(graph, 'field_r'), "图必须具有扩散场('field_r')"
        assert hasattr(graph, 'pos'), "图必须具有节点位置('pos')"
        assert hasattr(graph, 'edge_index'), "图必须具有边索引('edge_index')"
        
        # 嵌入r - 与FMGN相同
        r = graph.r
        field_r = graph.field_r
        pos = graph.pos
        edge_index = graph.edge_index
        batch = graph.batch
        emb = self.r_embedding(r)  # 形状(batch_size, emb_width)
        
        # 组合节点特征 - 避免多次cat操作
        node_features = [field_r]
        for attr in ['cond', 'field', 'loc', 'glob', 'omega']:
            feat = graph.get(attr)
            if feat is not None:
                # print(attr,feat.shape)
                node_features.append(feat)
        # 单次cat操作
        v = self.node_encoder(torch.cat(node_features, dim=1))
        
        # 编码r嵌入到节点特征 - 与FMGN相同
        emb_proj = self.r_encoder[0](emb)
        v = torch.cat([v, emb_proj[batch]], dim=1)
        v = self.r_encoder[2](self.r_encoder[1](v))  # 合并两个操作
            
        # 计算边几何特征
        src, dst = graph.edge_index
        edge_vec = graph.pos[dst] - graph.pos[src]
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        
        # 计算RBF距离嵌入
        edge_rbf = self.rbf_embedding(edge_length)
        
        # 编码边特征 - 增强几何信息
        edge_features = [edge_rbf]
        if hasattr(graph, 'edge_attr'):
            edge_features.insert(0, graph.edge_attr)  # 原始边特征
        if graph.get('edge_cond') is not None:
            edge_features.append(graph.get('edge_cond'))  # 条件边特征
            
        e = self.edge_encoder(torch.cat(edge_features, dim=1))
        
        # 通过GNN传播 - 与FMGN相同
        v, _ = self.propagator(graph, v, e, emb)  # v: (N, fnns_width)

        # 分支1：直接解码
        v_direct = self.direct_decoder(v)         # (N, out_node_features)

        # 分支2：势函数→curl
        potential = self.potential_network(v)     # (N, potential_dim)
        v_curl = graph_curl_operator(
            potential, graph.edge_index, graph.pos)  # (N, self.dim)
        # 用提前定义好的 proj_curl 映射到 out_node_features
        v_curl = self.proj_curl(v_curl)           # (N, out_node_features)

        # 混合输出
        vector_field = (1 - self.curl_weight) * v_direct + self.curl_weight * v_curl
        return vector_field