import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True,sigma = 5.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.size_average = size_average

    def forward(self, x0, x1, targets):
        s_diff = self.sigma * (x0 - x1)
        P = torch.cat((torch.sigmoid(s_diff),torch.sigmoid(-s_diff)),dim=1)
        #print(P.shape)
	predic = torch.argmax(P,1).view(-1,1)
	predic = predic * 2 - 1
        #print(predic,predic.size())
        N = P.size(0)
        #print(N)
        C = P.size(1)
        #print(C)

        class_mask = P.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
	targets = (targets.long() + 1) / 2
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        probs = (P*class_mask).sum(dim=1).view(-1,1)

        log_p = probs.log()

        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss,predic

        

if __name__ == "__main__":
    FL = FocalLoss(gamma=2 )
    N = 10
    C = 1
    x0 = torch.rand(N, C)
    x1 = torch.rand(N, C)
    print(x0,x1)
    targets = torch.LongTensor(N).random_(2)
    targets = targets * 2 - 1
    print('targets',targets)
    fl_loss,predic = FL(x0,x1, targets)
    print(fl_loss,fl_loss.shape,type(fl_loss))
    print('predic',predic)
    correct = (targets == predic).sum()
    print(correct)
 
 
