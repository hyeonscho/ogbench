import torch
import numpy as np

class VOGMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    '''
    Offline RL dataset for 2D maze environments from OG-Bench.
    Large
        Mean of obs: 139.58089505083132
        std of obs: 71.31185013523307
        Mean of pos: [16.702621 10.974173]
        std of pos: [10.050303   6.8203936]
    Giant
        Mean of obs: 141.01873851323037
        std of obs: 73.4250522212486
        Mean of pos: [24.888689 17.158426]
        std of pos: [14.732276 11.651127]
    '''
        
    def __init__(self, dataset_url='/home/hyeons/workspace/ogbench/ogbench/dataset/visual-pointmaze-medium-navigate-v0.npz' , split: str = "training"):
        
        super().__init__()
        self.dataset_url = dataset_url
        self.split = split
        dataset = self.get_dataset(self.dataset_url)
        self.observations = dataset["observations"]
        self.pos = dataset["qpos"]
        self.actions = dataset["actions"]

        # Normalizations
        if 'large' in dataset_url:
            self.observations = (self.observations - 139.58089505083132) / 71.31185013523307
        elif 'giant' in dataset_url:
            self.observations = (self.observations - 141.01873851323037) / 73.4250522212486

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.observations[idx]).float().permute(2, 0, 1)
        pos = torch.from_numpy(self.pos[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        return observation, pos, action
    
    def __len__(self):
        return len(self.observations)

    def get_dataset(self, path):
        if self.split == "validation":
            path = path.replace(".npz", "-val.npz")
        dataset = np.load(path, allow_pickle=True, mmap_mode='r')  # 메모리 매핑 적용
        return dataset

if __name__ == '__main__':
    dataset = VOGMaze2dOfflineRLDataset(dataset_url='/home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-large-navigate-v0.npz', split='training')
    print('large Mean of obs:', np.mean(dataset.observations))
    print('large std of obs:', np.std(dataset.observations))
    print("large Mean of pos:", np.mean(dataset.pos, axis=0))
    print("large std of pos:", np.std(dataset.pos, axis=0))

    dataset = VOGMaze2dOfflineRLDataset(dataset_url='/home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-giant-navigate-v0.npz', split='training')
    print('Giant Mean of obs:', np.mean(dataset.observations))
    print('Giant std of obs:', np.std(dataset.observations))
    print("Giant Mean of pos:", np.mean(dataset.pos, axis=0))
    print("Giant std of pos:", np.std(dataset.pos, axis=0))

