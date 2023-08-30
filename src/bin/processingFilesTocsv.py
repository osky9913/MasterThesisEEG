import torch
from torch.utils.data import Dataset
import mat73
PATH = "/datasets/Dasps.mat/S02preprocessed.mat"


class MatDataset(Dataset):
    def __init__(self, mat_file):
        self.data = list(mat73.loadmat(mat_file)["data"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X = torch.from_numpy(self.data[idx])
        return X
# Usage:


dataset = MatDataset(PATH)
print(len(dataset))
print(dataset[1])
print(dataset[1].shape())


"""
Questions :
1. How to get a y values ? 
2. "The participant is prepared to start the experiment, with closed eyes and minimizing gesture and speech.
    The psychotherapist starts by reciting the first situation and helps the subject imagining it.
    This phase is divided into two stages: recitation by the psychotherapist for the first 15 seconds and Recall by the subject for the last 15 seconds."
    15 seconds in the signal is how long vector from 1920"
3. Is datasets balanced?
4.    


Original paper:
https://arxiv.org/abs/1901.02942 ( it should be )

for matlab script for 6 situations ->  on the following site:
http://www.regim.org/publications/databases/dasps/

The paper for classyfing:
https://dl.acm.org/doi/pdf/10.1145/3486001.3486227
tentwenty placement of electrods 
"""
