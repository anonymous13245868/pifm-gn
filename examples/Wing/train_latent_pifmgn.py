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

torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------------------
# arguments
# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int, default=0)
parser.add_argument('--gpu',           type=int, default=0)
args = parser.parse_args()

seed = 0
torch.manual_seed(seed)

# ------------------------------------
experiment = {
    0: {
        'name':          'Latent_PIFMGN_Wing',
        'autoencoder':   './checkpoints/ae-nt250.chk', 
        'depths':        [1, 2, 2, 2],
        'width':         128,
        'nt':            250,
        'potential_dim': 3,
        'rbf_dim':       16,
    },
}[args.experiment_id]

# ------------------------------------
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    tensor_board  = './boards',
    chk_interval  = 10,
    training_loss = dgn.nn.losses.FlowMatchingLoss(),
    epochs        = 50000,
    batch_size    = 32,  
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 250, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# ------------------------------------
transform = transforms.Compose([
    dgn.transforms.ScaleEdgeAttr(0.015),
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    dgn.transforms.ScaleAttr('target', vmin=-1850, vmax=400),
    dgn.transforms.MeshCoarsening(
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True,
    ),
])

# ------------------------------------
dataset = dgn.datasets.pOnWing(
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
from ...nn.flow_matching.models.clfmgn_mix.latent_PIFMGN import LatentConservationFlowMatchingGraphNet

arch = {
    'dim':                3,
    'in_node_features':   1,
    'cond_node_features': 126,
    'cond_edge_features': 126,
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
    'curl_weight':        0.2,
    'div_penalty':        0.1,
}

model = LatentConservationFlowMatchingGraphNet(
    autoencoder_checkpoint = experiment['autoencoder'],
    arch                   = arch,
)

# ------------------------------------
model.fit(train_settings, dataloader)