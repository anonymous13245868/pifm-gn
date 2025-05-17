"""
    Train a Conservation Flow‐Matching Graph Net (PIFMGN) to predict the velocity (u,v)
    and pressure (p) fields around an ellipse.
    Run with:
        python train_pifmgn_ellipseflow.py --experiment_id 0 --gpu 0
"""

import os
import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn
# 保证多进程下共享内存不会出错
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------------------
# arguments
# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int, default=0)
parser.add_argument('--gpu',           type=int, default=0)
args = parser.parse_args()

# 固定随机种子
seed = 0
torch.manual_seed(seed)

# ------------------------------------
# experiment 配置
# ------------------------------------
experiment = {
    0: {
        'name':          'PIFMGN_EllipseFlow',
        'depths':        [2, 2, 2, 2, 2],  
        'width':         128,
        'nt':            10,  
        'potential_dim': 2,   
        'rbf_dim':       16,  
    },
}[args.experiment_id]

# ------------------------------------
# Training settings
# ------------------------------------
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.FlowMatchingLoss(),
    epochs        = 5000,
    batch_size    = 32,       
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 50, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# ------------------------------------
# 数据预处理 transform
# ------------------------------------
transform = transforms.Compose([
    dgn.transforms.ConnectKNN(6),                                                                              # Create an edge to each node from its 6 nearest neighbors
    dgn.transforms.ScaleEdgeAttr(0.15),                                                                        # Scale the edge attributes (relative positions)
    dgn.transforms.ScaleNs({'u': (-1.8,1.8), 'v': (-1.8,1.8), 'p': (-3, 3), 'Re': (500,1000)}, format='uvp'),  # Scale the node attributes 
    dgn.transforms.AddDirichletMask(3, [0,1], dirichlet_boundary_id=[2, 4]),                                   # Add a 3-component mask to indicate the Dirichlet boundary for each variable (u, v, p)
    dgn.transforms.MeshCoarsening(                                                                             # Create 4 lower-resolution graphs and normalise the relative position betwen the inter-graph nodes
        num_scales      =  5,
        rel_pos_scaling = [0.15, 0.3, 0.6, 1.2, 2.4],
        scalar_rel_pos  = True, 
    ),
])

# ------------------------------------
# 数据集与 DataLoader
# ------------------------------------
dataset = dgn.datasets.uvpAroundEllipse(
    # path      = dgn.datasets.DatasetDownloader(
    #                 dgn.datasets.DatasetUrl.uvpAroundEllipseTrain
    #             ).file_path,
    path      = '/PATH/TO/datasets--mariolinov--Ellipse/uvpAroundEllipseTrain.h5',
    T         = experiment['nt'],
    transform = transform,
    preload   = False,
)

dataloader = dgn.DataLoader(
    dataset            = dataset,
    batch_size         = train_settings['batch_size'],
    shuffle            = True,
    num_workers        = min(24, os.cpu_count()),
    pin_memory         = True,
    persistent_workers = True,
    prefetch_factor    = 4,
)

# ------------------------------------
# 构建模型
# ------------------------------------
from dgn4cfd.nn.flow_matching.models.pifmgn import ConservationFlowMatchingGraphNet_MIX

arch = {
    'in_node_features':   3,  
    'cond_node_features': 4,  # 条件特征：Re, d_inner, d_inlet, d_wall
    'cond_edge_features': 2,  
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
    # PIFMGN 专属
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
    'dim':                2,  # 2D 问题
}

model = ConservationFlowMatchingGraphNet_MIX(arch = arch)

# ------------------------------------
# 启动训练
# ------------------------------------
model.fit(train_settings, dataloader)