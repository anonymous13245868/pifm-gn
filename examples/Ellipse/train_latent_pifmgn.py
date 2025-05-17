"""
    Train a Latent Conservation Flow-matching Graph Net (LPIFMGN) to predict the pressure field on an ellipse.
    Run with:
        python train_lpifmgn.py --experiment_id 0 --gpu 0
"""
import os
import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn

torch.multiprocessing.set_sharing_strategy('file_system')

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--gpu',           type=int, default=0)
args = argparser.parse_args()

# Initial seed
seed = 0
torch.manual_seed(seed)

# Dictionary of experiments
experiment = {
    0: {
        'name':         'Latent_PIFMGN_Ellipse',
        'autoencoder':  './checkpoints/ae-nt10.chk', 
        'depths':       [1,2],  
        'width':        128,
        'nt':           10,    
        'potential_dim': 1,    
        'rbf_dim':      16,   
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints_my',
    # checkpoint  = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 10,
    training_loss = dgn.nn.losses.FlowMatchingLoss(),  
    epochs        = 5000,
    batch_size    = 64,  
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.5, "patience": 50, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset - 确保包含位置信息和适合VGAE的变换
transform = transforms.Compose([
    dgn.transforms.MeshEllipse(),                               
    dgn.transforms.ScaleEdgeAttr(0.02),                        
    dgn.transforms.EdgeCondFreeStreamProjection(),              
    dgn.transforms.ScaleAttr('target', vmin=-1.05, vmax=0.84),  
    dgn.transforms.ScaleAttr('glob',   vmin=500,   vmax=1000),  
    dgn.transforms.ScaleAttr('loc',    vmin=2,     vmax=3.5),   
    dgn.transforms.MeshCoarsening(                             
        num_scales      =  4,
        rel_pos_scaling = [0.02, 0.06, 0.15, 0.3],
        scalar_rel_pos  = True, 
    ),
])

dataset = dgn.datasets.pOnEllipse(
    # path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnEllipseTrain).file_path,
    path      = '/PATH/TO/datasets--mariolinov--Ellipse/pOnEllipseTrain.h5',
    T         = experiment['nt'],
    transform = transform,
    preload   = True,
)

dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = min(8, os.cpu_count()),   
    pin_memory  = True,                     
    persistent_workers = True,              
    prefetch_factor = 2,                    
)

arch = {
    'in_node_features':   1,    
    'cond_node_features': 128,  
    'cond_edge_features': 128,  
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

from dgn4cfd.nn.flow_matching.models.latent_PIFMGN import LatentConservationFlowMatchingGraphNet

model = LatentConservationFlowMatchingGraphNet(
    autoencoder_checkpoint = experiment['autoencoder'],
    arch = arch,
)

# 训练
model.fit(train_settings, dataloader)