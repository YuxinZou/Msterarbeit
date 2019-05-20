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
        #print(s_diff)
        loss = (1.0 + label) * s_diff / 2.0  - F.logsigmoid(s_diff+1e-10)
        #loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.sum(loss) / x0.size()[0]

        return loss

if __name__ == "__main__":
    loss = CrossEntropy()
    a = torch.randn(10,1)
    b = torch.randn(10,1)
    label = torch.zeros(10)
    print(a-b)
    print(loss.forward(a,b,label))
