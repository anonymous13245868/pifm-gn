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
from ...nn.flow_matching.models.clfmgn_mix import ConservationFlowMatchingGraphNet_MIX

torch.multiprocessing.set_sharing_strategy('file_system')

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--gpu',           type=int, default=0)
args = argparser.parse_args()

seed = 0
torch.manual_seed(seed)

experiment = {
    0: {
        'name':          'PIFMGN_Wing',
        'nt':             250,   
        'potential_dim':   3,     
        'rbf_dim':        16,     
    },
}[args.experiment_id]

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

transform = transforms.Compose([
    dgn.transforms.ScaleEdgeAttr(0.015),
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    dgn.transforms.ScaleAttr('target', vmin=-1850, vmax=400),
    dgn.transforms.MeshCoarsening(
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True,
    ),
    # dgn.transforms.PreserveAttribute('pos'),
])

dataset = dgn.datasets.pOnWing(
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

arch = {
    'dim':                3,   
    'in_node_features':   1,   
    'cond_node_features': 3,   
    'cond_edge_features': 6,   
    'depths':             6 * [2],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
}

model = ConservationFlowMatchingGraphNet_MIX(arch=arch)

# 启动训练
model.fit(train_settings, dataloader)