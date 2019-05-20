import torch
import torch.nn as nn
import torch.nn.functional as F
from PointNetFeat import PointNetfeat
from RankNet import RankNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() 
        self.feature = nn.Sequential(
            PointNetfeat(),
	    RankNet(),
        )
        
    def forward(self, x):
        out = self.feature(x)
        return out



if __name__ == "__main__":
    model = Model()
    print(model(torch.randn(1,2,2048)))
