
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.vog_maze_emb import VOGEmbeddingDataset
from models.mlp import MLP
from tqdm import tqdm
import argparse

# configs
parser = argparse.ArgumentParser(description='Training 3 layer MLP to predict the state from the embedding')

parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--name', type=str, default='beta_vae', help='Model name')
parser.add_argument('--load', type=str, default=None, help='Path to load model')
parser.add_argument('--group_name', type=str, default='None', help='gropu name')
parser.add_argument('--log', type=bool, default=True, help='log')
parser.add_argument('--hidden_dim', type=int, default=1024, help='log')
parser.add_argument('--num_layers', type=int, default=3, help='log')
parser.add_argument('--dataset_url', type=str, default='None', help='dataset url')

args = parser.parse_args()
config = vars(args)


# set seed
torch.manual_seed(config["seed"])
if config["device"] == "cuda":
    torch.cuda.manual_seed(config["seed"])

#initialize wandb

if config["log"]:
    import wandb
    wandb.init(project='hs_e2s', entity='Hierarchical-Diffusion-Forcing', config=config)

# Load the dataset
train_dataset = VOGEmbeddingDataset(config['dataset_url'], split='training')
validation_dataset = VOGEmbeddingDataset(config['dataset_url'], split='validation')

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False)

print("Dataset loaded")

# Initialize the model
model = MLP(input_dim=8, output_dim=2, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'])
model.to(config["device"])

# Load the model if specified
if config["load"]:
    model.load_state_dict(torch.load(config["load"]))

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Train
print("Training started")
for epoch in tqdm(range(config["num_epochs"])):
    model.train()
    train_loss_list = []
    for i, (emb, pos, _) in enumerate(train_loader):
        emb, pos = emb.to(config["device"]), pos.to(config["device"])
        optimizer.zero_grad()
        pred = model(emb)
        loss = nn.functional.mse_loss(pred, pos)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.mean().item())

    # Validation
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            loss_list = []
            for i, (emb, pos, _) in enumerate(validation_loader):            
                emb, pos = emb.to(config["device"]), pos.to(config["device"])
                pred = model(emb)
                loss = nn.functional.mse_loss(pred, pos)
                loss_list.append(loss.mean().item())
            print(f"Epoch: {epoch}, Train Loss: {sum(train_loss_list)/len(train_loss_list)}, Validation Loss: {sum(loss_list)/len(loss_list)}")                
            
            if config["log"]:
                wandb.log({"Training Loss": sum(train_loss_list)/len(train_loss_list)})
                wandb.log({"Validation Loss": sum(loss_list)/len(loss_list)})

    # Save the model with time
    # if epoch % 10 == 0:
    #     torch.save(model.state_dict(), f"e2s_loss{sum(loss_list)/len(loss_list)}_{epoch}.pth")
maze_type = 'large' if 'large' in config['dataset_url'] else 'giant'
torch.save(model.state_dict(), f"{maze_type}_e2s.pth")
