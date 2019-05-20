import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        print(N)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        print(P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
	print('(class_mask)',class_mask)
        ids = targets.view(-1, 1)
	print('ids',ids)
        class_mask.scatter_(1, ids.data, 1.)
        print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
	print(alpha)
        print('P',P)
        probs = (P*class_mask).sum(dim=1).view(-1,1)
	print('probs',probs)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

        

if __name__ == "__main__":
    alpha = torch.rand(21, 1)
    print(alpha)
    FL = FocalLoss(class_num=5, gamma=2 )
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 5
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    print(targets)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    print(fl_loss,fl_loss.shape,type(fl_loss))
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl ={}'.format(ce_loss.data, fl_loss.data))
    fl_loss.backward()
    ce_loss.backward()
    #print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)
 
