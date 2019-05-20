import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self):
	super(RankNet, self).__init__() 
        self.ranknet = nn.Sequential(
            nn.Linear(2048, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            #nn.LeakyReLU(0.2, inplace=True),    
        )  

    def forward(self, x):
        out = self.ranknet(x)
        return out

if __name__ == "__main__":
    model = RankNet()
    print(model(torch.randn(2048)))
