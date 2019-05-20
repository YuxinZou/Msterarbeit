import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self):
	super(RankNet, self).__init__() 
        self.ranknet = nn.Sequential(
            nn.Linear(1024, 512),
	    #nn.BatchNorm1d(512),
	    #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 128),
	    #nn.BatchNorm1d(128),
	    #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),   

            nn.Linear(128, 1),
        )  

    def forward(self, x):
        out = self.ranknet(x)
        return out

if __name__ == "__main__":
    model = RankNet()
    a = torch.randn(1,1024)
    print(model(a))
