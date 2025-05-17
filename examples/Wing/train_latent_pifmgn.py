"""
    Train a Latent Conservation Flow‐Matching Graph Net (LPIFMGN) to predict the pressure field on a wing
    via a latent flow‐matching model.
    Run with:
        python train_latent_pifmgn_wing.py --experiment_id 0 --gpu 0
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
        'name':          'Latent_PIFMGN_Wing',
        # 自编码器检查点，请替换为你的 AE 模型路径
        'autoencoder':   './checkpoints/ae-nt250.chk',  # 请替换为自编码器检查点的实际路径
        # 在 latent 空间中使用的 GNN 深度（可以根据需求调整）
        'depths':        [1, 2, 2, 2],
        'width':         128,
        # 最长模拟时间步数
        'nt':            250,
        # 势函数维度（3D 问题中通常取 3）
        'potential_dim': 3,
        # RBF 距离嵌入维度
        'rbf_dim':       16,
    },
}[args.experiment_id]

# ------------------------------------
# 训练设置
# ------------------------------------
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    tensor_board  = './boards',
    chk_interval  = 10,
    # 使用物理信息流匹配损失
    training_loss = dgn.nn.losses.FlowMatchingLoss(),
    epochs        = 50000,
    batch_size    = 32,  # 视显存情况调整
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 250, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# ------------------------------------
# 数据预处理 transform
# ------------------------------------
transform = transforms.Compose([
    # 缩放边上的相对位置向量
    dgn.transforms.ScaleEdgeAttr(0.015),
    # 将自由流方向（法向量）加入边条件
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    # 缩放目标场（压力 p）
    dgn.transforms.ScaleAttr('target', vmin=-1850, vmax=400),
    # 网格下采样：6 级粗图
    dgn.transforms.MeshCoarsening(
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True,
    ),
])

# ------------------------------------
# 数据集 & DataLoader
# ------------------------------------
dataset = dgn.datasets.pOnWing(
    # 使用官方下载器，也可替换成本地路径
    path = '/PATH/TO/datasets--mariolinov--Wing/pOnWingTrain.h5',
    T         = experiment['nt'],
    transform = transform,
    preload   = True,
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

# ------------------------------------
# 构建模型
# ------------------------------------
from dgn4cfd.nn.flow_matching.models.latent_PIFMGN import LatentConservationFlowMatchingGraphNet

arch = {
    # 物理坐标维度（3D 问题）
    'dim':                3,
    # 输入噪声维度：潜在 p 的一维高斯噪声
    'in_node_features':   1,
    # 自编码器编码后的节点潜空间维度（+ 可选附加 cond, 此处为 126）
    'cond_node_features': 126,
    # 自编码器编码后的边潜空间维度
    'cond_edge_features': 126,
    # GNN 深度与宽度
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
    # Conservation‐FlowMatching 专属
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
    # 旋度分支和散度惩罚权重
    'curl_weight':        0.2,
    'div_penalty':        0.1,
}

model = LatentConservationFlowMatchingGraphNet(
    autoencoder_checkpoint = experiment['autoencoder'],
    arch                   = arch,
)

# ------------------------------------
# 启动训练
# ------------------------------------
model.fit(train_settings, dataloader)