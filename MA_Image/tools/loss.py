import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, margin=1.0):
        super(CrossEntropy, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, label):  
        # euclidian distance
        s_diff = x0 - x1
        loss = (1.0 + label).mul(s_diff) /2.0  - F.logsigmoid(s_diff+1e-10)
	#loss =  label.mul(s_diff) - F.logsigmoid(s_diff+1e-10)
        #loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        #loss = torch.sum(loss) / x0.size()[0]
	print(loss)
	#print(loss.shape)
	loss = torch.mean(loss,dim=0,keepdim=False)
        return loss

if __name__ == "__main__":
    loss = CrossEntropy()
    #a = torch.randn(10,1)
    #b = torch.randn(10,1)
    a = torch.FloatTensor([[0.9067],
        [0.2650],
        [0.0798],
        [0.0718],
        [0.5946],
        [0.7905],
        [0.7696],
        [0.7743],
        [0.1238],
        [0.8600]])
    b = torch.FloatTensor([[0.2506],
        [0.7927],
        [0.6980],
        [0.3326],
        [0.8282],
        [0.9184],
        [0.8756],
        [0.0530],
        [0.0661],
        [0.0386]])
    label = torch.Tensor([ 1, -1, -1, -1,  1,  1,  1, -1,  1, -1]).view(-1,1)
    #print(a,b,label)
    print(loss.forward(a,b,label))
