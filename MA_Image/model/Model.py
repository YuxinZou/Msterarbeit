import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet18 import ResNet18
from ResNet34 import ResNet34
from RankNet import RankNet
from Inception import Inception3
from residual_attention_network import ResidualAttentionModel_56
class Model(nn.Module):
    def __init__(self, layers = 18):
        super(Model, self).__init__() 
	self.layers = layers
	if self.layers == 34:
	    resnet = ResNet34()
	else:
	    resnet = ResNet18()

        self.feature = nn.Sequential(
            #resnet,
            ResidualAttentionModel_56(),
	    RankNet(),
        )
        
    def forward(self, x):
        out = self.feature(x)
        #print(out)
        out = torch.sigmoid(out)
        return out



if __name__ == "__main__":
    model = Model()
    print(model(torch.randn(10,1,256,256)))
