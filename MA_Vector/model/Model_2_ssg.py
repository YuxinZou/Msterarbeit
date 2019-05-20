import torch
import torch.nn as nn
import torch.nn.functional as F
from PointNetFeat2 import PointNet2ClsSsg
from RankNet import RankNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() 
        self.feature = nn.Sequential(
            PointNet2ClsSsg(),
	    RankNet(),
        )
        
    def forward(self, x):
        out = self.feature(x)
        out = torch.sigmoid(out)
        return out



if __name__ == "__main__":
    model = Model()
    print(model(torch.randn(32,2,2048)))
