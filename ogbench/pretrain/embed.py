'''
This file is used to embed the data into the latent space using the pre-trained VAE model.
command
CUDA_VISIBLE_DEVICES=2 python embed.py
'''

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.vog_maze import VOGMaze2dOfflineRLDataset
from models.bvae import BetaVAE
from tqdm import tqdm
import argparse

# configs
parser = argparse.ArgumentParser(description='Beta VAE Training Configuration')
parser.add_argument('--dataset_url', type=str, default='None', help='dataset url')
parser.add_argument('--load', type=str, default=None, help='Path to load model')
args = parser.parse_args()
config = vars(args)
# Load Model
path_to_model = config['load']
model = BetaVAE().cuda()
model.load_state_dict(torch.load(path_to_model))
model.eval()
print("Model loaded successfully")

maze_type = 'large' if 'large' in path_to_model else 'giant'
# Load Dataset
path_to_dataset  = config['dataset_url']
splits = ['training', 'validation'] 
for split in splits:
    dataset = VOGMaze2dOfflineRLDataset(path_to_dataset, split) 
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
    print("Dataset loaded successfully")
    print(dataset.observations.shape)
    # Embed data and save in chunks
    path_to_save = './embedded_data/' + maze_type + split
    os.makedirs(path_to_save, exist_ok=True)

    embeddings = []
    positions = []
    actions = []
    chunk_size = 10000  # Adjust the chunk size as needed
    for i, (obs, pos, act) in enumerate(tqdm(dataloader, desc="Embedding data")):
        with torch.no_grad():
            obs = obs.cuda()
            mean, log_var = model.encode(obs)
            z = model.reparameterize(mean, log_var)
            embeddings.append(z.cpu().numpy())
            positions.append(pos.cpu().numpy())
            actions.append(act.cpu().numpy())
        

    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        print(embeddings.shape)
        positions = np.concatenate(positions, axis=0)
        actions = np.concatenate(actions, axis=0)
        with open(os.path.join(path_to_save, f'latent_{split}.npy'), 'ab') as f:
            np.save(f, embeddings)
        with open(os.path.join(path_to_save, f'positions_{split}.npy'), 'ab') as f:
            np.save(f, positions)
        with open(os.path.join(path_to_save, f'actions_{split}.npy'), 'ab') as f:
            np.save(f, actions)