import torch
from torch.utils.data import DataLoader
from ogbench.pretrain.dataset.vog_maze import VOGMaze2dOfflineRLDataset
from ogbench.pretrain.models.bvae import BetaVAE
from tqdm import tqdm
import argparse

# configs
parser = argparse.ArgumentParser(description='Beta VAE Training Configuration')

parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--name', type=str, default='beta_vae', help='Model name')
parser.add_argument('--load', type=str, default=None, help='Path to load model')
parser.add_argument('--kld_weight', type=float, default=0.1, help='KLD weight')
parser.add_argument('--group_name', type=str, default='None', help='gropu name')
parser.add_argument('--dataset_url', type=str, default='None', help='dataset url')

args = parser.parse_args()
config = vars(args)


# set seed
torch.manual_seed(config["seed"])
if config["device"] == "cuda":
    torch.cuda.manual_seed(config["seed"])

#initialize wandb
import wandb
wandb.init(project='hs_vae', entity='Hierarchical-Diffusion-Forcing', config=config)

# Load the dataset
train_dataset = VOGMaze2dOfflineRLDataset(dataset_url=config['dataset_url'], split='training')
validation_dataset = VOGMaze2dOfflineRLDataset(dataset_url=config['dataset_url'], split='validation')

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False)

print("Dataset loaded")

# Initialize the model
model = BetaVAE()
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
    for i, (obs, pos, _) in enumerate(train_loader):
        obs = obs.to(config["device"])
        optimizer.zero_grad()
        recons, inputs, mu, log_var = model(obs)
        loss_dict = model.loss_function(recons, inputs, mu, log_var)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iter: {model.num_iter}", end=" " )
            for k, v in loss_dict.items():
                wandb.log({k: v})
                print(f"{k}: {v}", end=", ")
            print(end="\r")
        
    # Validation
    model.eval()
    with torch.no_grad():
        for i, (obs, pos, _) in enumerate(validation_loader):            
            obs = obs.to(config["device"])
            recons, inputs, mu, log_var = model(obs)
            loss_dict = model.loss_function(recons, inputs, mu, log_var)
            if i == 0:
                idx = torch.randint(0, obs.size(0), (32,))
                # log the base input and reconstruction
                inputs = inputs[idx[:8]]
                recons = recons[idx[:8]]
                inputs = inputs * 71.0288272312382 + 141.785487953533
                recons = recons * 71.0288272312382 + 141.785487953533
                wandb.log({"input": wandb.Image(inputs)})
                wandb.log({"recons": wandb.Image(recons)})

                
    for k, v in loss_dict.items():
        wandb.log({k+'val': v})
        print("Epoch:", epoch,k+'val', ":", v, end=", ")

    # Save the model with time
    if epoch % 10 == 0:
        maze_type = 'large' if 'large' in config['dataset_url'] else 'giant'
        torch.save(model.state_dict(), f"{maze_type}_loss{loss_dict['Reconstruction_Loss']}_{epoch}.pth")
torch.save(model.state_dict(), f"{maze_type}_loss{loss_dict['Reconstruction_Loss']}_last.pth")



