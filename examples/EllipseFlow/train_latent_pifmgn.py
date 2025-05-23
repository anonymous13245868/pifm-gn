"""
    Train a Latent Conservation Flow‐Matching Graph Net (LPIFMGN) to predict the velocity (u,v)
    and pressure (p) fields around an ellipse in latent space.
    Run with:
        python train_latent_pifmgn_ellipseflow.py --experiment_id 0 --gpu 0
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
        'name':          'Latent_PIFMGN_EllipseFlow',
        'autoencoder':   './checkpoints/ae-nt10.chk',  #
        'depths':        [1, 2, 2],  #
        'width':         128,
        'nt':            10,  # 
        'potential_dim': 2,   #
        'rbf_dim':       16,  #
    },
}[args.experiment_id]

# ------------------------------------
# Training settings
# ------------------------------------
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    tensor_board  = './boards',
    chk_interval  = 10,
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
dataset = dgn.datasets.uvpAroundEllipse(
    path      = '/PATH/TO/datasets--mariolinov--Ellipse/uvpAroundEllipseTrain.h5',
    T         = experiment['nt'],
    transform = transform,
    preload   = False,
)

dataloader = dgn.DataLoader(
    dataset            = dataset,
    batch_size         = train_settings['batch_size'],
    shuffle            = True,
    num_workers        = min(32, os.cpu_count()),
    pin_memory         = True,
    persistent_workers = True,
    prefetch_factor    = 4,
)

# ------------------------------------
from ...nn.flow_matching.models.latent_PIFMGN import LatentConservationFlowMatchingGraphNet

arch = {
    'in_node_features':   1,      
    'cond_node_features': 126,    
    'cond_edge_features': 126,    
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
    'potential_dim':      experiment['potential_dim'],
    'rbf_dim':            experiment['rbf_dim'],
    'dim':                2,      
    'curl_weight':        0.2,   
    'div_penalty':        0.1,  
}

model = LatentConservationFlowMatchingGraphNet(
    autoencoder_checkpoint = experiment['autoencoder'],
    arch = arch,
)

# ------------------------------------
model.fit(train_settings, dataloader)