"""
    Train a Conservation Flow-matching Graph Net (PIFMGN) to predict the pressure field on a wing.
    Run with:
        python train_pifmgn_pOnWing.py --experiment_id 0 --gpu 0
"""

import os
import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn
# 引入 PIFMGN 模型
from dgn4cfd.nn.flow_matching.models.clfmgn_mix import ConservationFlowMatchingGraphNet_MIX

torch.multiprocessing.set_sharing_strategy('file_system')

# 解析命令行参数
argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--gpu',           type=int, default=0)
args = argparser.parse_args()

# 固定随机种子，确保可重复
seed = 0
torch.manual_seed(seed)

# 不同实验配置
experiment = {
    0: {
        'name':          'PIFMGN_Wing',
        'nt':             250,   # 时序长度与 FMGN 相同
        'potential_dim':   3,     # 3D 流场对应 3 维势函数
        'rbf_dim':        16,     # RBF 距离嵌入维度
    },
}[args.experiment_id]

# 训练超参数
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    # checkpoint  = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.FlowMatchingLoss(),
    epochs        = 5000,
    batch_size    = 32,
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 250, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# 数据预处理 transform
transform = transforms.Compose([
    # 缩放边上相对位置向量
    dgn.transforms.ScaleEdgeAttr(0.015),
    # 增加自由流速度在局部边坐标系下的投影
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    # 缩放目标场（压力）
    dgn.transforms.ScaleAttr('target', vmin=-1850, vmax=400),
    # 多尺度图粗化
    dgn.transforms.MeshCoarsening(
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True,
    ),
    # （可选）如果后续需要显式 pos 属性，可加上：
    # dgn.transforms.PreserveAttribute('pos'),
])

# 数据集及加载器
dataset = dgn.datasets.pOnWing(
    # path       = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnWingTrain).file_path,
    path = '/PATH/TO/datasets--mariolinov--Wing/pOnWingTrain.h5',
    T          = experiment['nt'],
    transform  = transform,
    preload    = False,
)
dataloader = dgn.DataLoader(
    dataset            = dataset,
    batch_size         = train_settings['batch_size'],
    shuffle            = True,
    num_workers        = min(16, os.cpu_count()),
    pin_memory         = True,
    persistent_workers = True,
    prefetch_factor    = 4,
)

# PIFMGN 模型架构
arch = {
    'dim':                3,   # 3D 问题
    'in_node_features':   1,   # 噪声压力
    'cond_node_features': 3,   # 法向量 (nx, ny, nz)
    'cond_edge_features': 6,   # 边上相对位置 + 自由流投影 (共 6 维)
    'depths':             6 * [2],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
    # Conservation FlowMatching 特有
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
}

model = ConservationFlowMatchingGraphNet_MIX(arch=arch)

# 启动训练
model.fit(train_settings, dataloader)