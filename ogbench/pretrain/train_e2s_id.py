import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.vog_maze_emb import VOGEmbeddingDataset
from models.mlp import MLP
from tqdm import tqdm
import argparse
import numpy as np
from collections import deque

# configs
parser = argparse.ArgumentParser(description='Training MLP on VOG embeddings')
parser.add_argument('--mode', type=str, choices=['invd', 'e2s'], required=True, help='Training mode: invd or e2s')
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--name', type=str, default='mlp_model')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--group_name', type=str, default='default_group')
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--frame_stack', type=int, default=3)
parser.add_argument('--dataset_url', type=str, default='None')
args = parser.parse_args()
config = vars(args)

# Set seed
torch.manual_seed(config["seed"])
if config["device"] == "cuda":
    torch.cuda.manual_seed(config["seed"])

# Initialize wandb
if config["log"]:
    import wandb
    wandb.init(project=f'{config["mode"]}_project', entity='Hierarchical-Diffusion-Forcing', config=config)

# Load raw datasets
train_raw = VOGEmbeddingDataset(config['dataset_url'], split='training')
val_raw = VOGEmbeddingDataset(config['dataset_url'], split='validation')

# Preprocessing
if config["mode"] == 'invd':
    def preprocess_inverse(dataset):
        processed = []
        emb_q = deque(maxlen=config["frame_stack"])
        actions = deque(maxlen=2)
        for i in range(len(dataset)):
            emb, _, action = dataset[i]
            emb_q.append(emb)
            actions.append(action)
            if i % 1001 == 0:
                if i == 0:
                    emb_q.append(emb)
                    continue
                else:
                    emb_q.append(emb)
            if len(emb_q) == config["frame_stack"]:
                processed.append((np.concatenate(list(emb_q)), actions[0]))
        return processed

    train_data = preprocess_inverse(train_raw)
    val_data = preprocess_inverse(val_raw)

    input_dim = 8 * config["frame_stack"]
    output_dim = 2
else:  # e2s
    train_data = [(emb, pos) for emb, pos, _ in train_raw]
    val_data = [(emb, pos) for emb, pos, _ in val_raw]
    input_dim = 8
    output_dim = 2

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

# Initialize model
model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'])
model.to(config["device"])

# Load weights
if config["load"]:
    model.load_state_dict(torch.load(config["load"]))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# Training loop
print("Training started")
for epoch in tqdm(range(config["num_epochs"])):
    model.train()
    train_loss_list = []
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(config["device"])
        targets = targets.to(config["device"])

        optimizer.zero_grad()
        preds = model(inputs)
        loss = nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())

    # Validation
    if epoch % 5 == 0:
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(config["device"])
                targets = targets.to(config["device"])
                preds = model(inputs)
                loss = nn.functional.mse_loss(preds, targets)
                val_loss_list.append(loss.item())

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if config["log"]:
            wandb.log({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})

# Save final model
maze_type = 'large' if 'large' in config['dataset_url'] else 'giant'
import datetime
date = datetime.datetime.today().strftime('%m%d')
torch.save(model.state_dict(), f"{maze_type}_{config['mode']}_{date}.pth")