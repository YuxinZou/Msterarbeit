import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
	#self.conv1 = nn.Conv1d(2, 64, 1)
	#self.conv2 = nn.Conv1d(64, 128, 1)
	#self.conv3 = nn.Conv1d(128, 1024, 1)
        #self.bn1 = nn.BatchNorm1d(64)
	#self.bn2 = nn.BatchNorm1d(128)
	#self.bn3 = nn.BatchNorm1d(1024)
        self.mlp1 = nn.Sequential(
		nn.Conv1d(2,64,1),
		nn.BatchNorm1d(64),
		nn.ReLU(),
		nn.Conv1d(64,64,1),
		nn.BatchNorm1d(64),
		nn.ReLU())
	self.mlp2 = nn.Sequential(
		nn.Conv1d(64,64,1),
		nn.BatchNorm1d(64),
		nn.ReLU(),
		nn.Conv1d(64,128,1),
		nn.BatchNorm1d(128),
		nn.ReLU(),
		nn.Conv1d(128,1024,1),
		nn.BatchNorm1d(1024),
                nn.ReLU())

    def forward(self, x):
    #batch size, channel , nummber of point
        batchsize = x.shape[0]
        n_pts = x.shape[2]
	#x = F.relu(self.bn1(self.conv1(x)))
	#x = F.relu(self.bn2(self.conv2(x)))
	#x = self.bn3(self.conv3(x))
	x = self.mlp1(x)
	x = self.mlp2(x)
	#return the maximum value of each row of the input tensor in the given dimension
	#x = torch.max(x, 2, keepdim=True)[0]
	x= F.max_pool1d(x,n_pts).squeeze(2)
#	x = x.view(-1, 1024)
	return x

 

if __name__ == "__main__":
    model = PointNetfeat()
    a = torch.randn(32, 2, 2048)
    print(model(a).shape)
