import torch
import numpy as np
import os

class VOGEmbeddingDataset(torch.utils.data.Dataset):
    '''
    Medium
    - pos mean: [10.273524  9.648321]
    - pos std: [5.627576 4.897987]
    - Action mean: [-0.00524961 -0.00168911]
    - Action std:  [0.70124096 0.6971626]
    Large
        emb mean: [ 0.56942457 -0.7748614   0.03422518 -0.05451093  0.00234696 -0.4766496
 -0.53628725 -1.1162051 ]
        emb std: [2.0956497 2.2737527 2.3882532 2.6977062 2.1805387 2.6994274 2.4300833
 2.137858 ]
        pos mean: [16.702621 10.974173]
        pos std: [10.050303   6.8203936]
        actions mean: [-0.01116096  0.00125011]
        actions std: [0.7068106 0.6878459]
    Giant
        emb mean: [ 0.67540884 -0.6614879  -0.30717567  0.09488879 -0.27652553 -0.8268671
        -1.1487181  -0.662139  ]
        emb std: [2.1224945 2.1889937 2.3098729 2.5563455 2.3711634 2.4826827 2.3423505
        2.6387706]
        pos mean: [24.888689 17.158426]
        pos std: [14.732276 11.651127]
        actions mean: [-0.00714872 -0.00213099]
        actions std: [0.70283055 0.69673675]
    '''
        
    def __init__(self, dataset_url='/home/hyeons/workspace/HierarchicalDiffusionForcing/data/embedded_data' , split: str = "training"):
        
        super().__init__()
        self.dataset_url = dataset_url
        self.split = split
        self.emb, self.pos, self.actions = self.get_dataset()

        # Normalizations
        if 'medium' in dataset_url:
            self.emb_mean = np.array([0.53332597, -0.57663816, -0.15480594, -0.10989726,  0.13822828, -0.7565398 , -0.67368555, -0.5261524])
            self.emb_std = np.array([2.230295 , 1.8695153, 2.5765393, 2.5024776, 2.409886 , 2.3264396, 2.2680814, 2.1177504])
            self.pos_mean = np.array([10.273524, 9.648321])
            self.pos_std = np.array([5.627576, 4.897987])
            self.actions_mean = np.array([-0.00524961, -0.00168911])
            self.actions_std = np.array([0.70124096, 0.6971626])
        elif 'large' in dataset_url:
            self.emb_mean = np.array([0.56942457, -0.7748614, 0.03422518, -0.05451093, 0.00234696, -0.4766496, -0.53628725, -1.1162051])
            self.emb_std = np.array([2.0956497, 2.2737527, 2.3882532, 2.6977062, 2.1805387, 2.6994274, 2.4300833, 2.137858])
            self.pos_mean = np.array([16.702621, 10.974173])
            self.pos_std = np.array([10.050303, 6.8203936])
            self.actions_mean = np.array([-0.01116096, 0.00125011])
            self.actions_std = np.array([0.7068106, 0.6878459])
        elif 'giant' in dataset_url:
            self.emb_mean = np.array([ 0.67540884, -0.6614879,  -0.30717567,  0.09488879, -0.27652553, -0.8268671, -1.1487181,  -0.662139])
            self.emb_std = np.array([2.1224945, 2.1889937, 2.3098729, 2.5563455, 2.3711634, 2.4826827, 2.3423505, 2.6387706])
            self.pos_mean = np.array([24.888689, 17.158426])
            self.pos_std = np.array([14.732276, 11.651127])
            self.actions_mean = np.array([-0.00714872, -0.00213099])
            self.actions_std = np.array([0.70283055, 0.69673675])

        # self.emb = (self.emb - self.emb_mean) / self.emb_std
        self.pos = (self.pos - self.pos_mean) / self.pos_std
        # self.actions = (self.actions - self.actions_mean) / self.actions_std


    def __getitem__(self, idx):
        emb = torch.from_numpy(self.emb[idx]).float()
        pos = torch.from_numpy(self.pos[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        return emb, pos, action
    
    def __len__(self):
        return len(self.emb)

    def get_dataset(self):
        path = os.path.join(self.dataset_url, self.split)
        emb = np.load(os.path.join(path, 'latent.npy'))
        pos = np.load(os.path.join(path, 'positions.npy'))
        act = np.load(os.path.join(path, 'actions.npy'))
        return emb, pos, act
if __name__ == '__main__':
    dataset = VOGEmbeddingDataset(dataset_url='/home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/large', split='training')
    print('large Mean of emb:', np.mean(dataset.emb, axis=0))
    print('large std of emb:', np.std(dataset.emb, axis=0))
    print("large Mean of pos:", np.mean(dataset.pos, axis=0))
    print("large std of pos:", np.std(dataset.pos, axis=0))
    print("large Mean of actions:", np.mean(dataset.actions, axis=0))
    print("Medium std of actions:", np.std(dataset.actions, axis=0))

    dataset = VOGEmbeddingDataset(dataset_url='/home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/giant', split='training')
    print('Giant Mean of emb:', np.mean(dataset.emb, axis=0))
    print('Giant std of emb:', np.std(dataset.emb,  axis=0))
    print("Giant Mean of pos:", np.mean(dataset.pos, axis=0))
    print("Giant std of pos:", np.std(dataset.pos, axis=0))
    print("Giant Mean of actions:", np.mean(dataset.actions, axis=0))
    print("Giant std of actions:", np.std(dataset.actions, axis=0))
