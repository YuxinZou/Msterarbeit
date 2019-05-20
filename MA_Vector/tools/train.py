import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.utils import data
from torchvision import transforms as T
import torch.utils.model_zoo as model_zoo
import os
from PIL import Image
import csv
import sys
from tensorboardX import SummaryWriter
sys.path.append("..")
from model.Model_2 import Model
from data.data_loader_process import TrainingSets,TestSets
from tools.loss import CrossEntropy
from tools.utils import init_weights

def train():
    vector_training_data_path = '../training_set_preprocess_2048/'
    csv_file_path = '../result_vector_with_0_50%.csv'
    vector_test_data_path ='../test_set_preprocess_2048/'
    
    #device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
#*****************Step1 Setup Data***********************
#load trainingset
    train_set = TrainingSets(csv_file_path,vector_training_data_path,2048) 
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True,num_workers=2)
#load testset
    test_set = TestSets(vector_test_data_path,2048)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=2)

#*****************Step2 Setup Model***********************           
    model = Model()
    model.to(device)
   # model = nn.DataParallel(model,device_ids=[0]).cuda()
    init_weights(model)

#*****************Step3 Setup Optimizer and Loss*********************** 
# loss function
    loss_func = CrossEntropy()
    loss_MSE = nn.MSELoss()

# optimizer
    #optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=0.005,betas=(0.9,0.999))  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    EPOCH = 20
# set tensor-board for visualization
   # writer = SummaryWriter('runs')
    #dummy_input =Variable(torch.rand(48, 2, 2048),requires_grad = True)
    #with SummaryWriter(comment='Net') as w:    
    #    w.add_graph(model,(dummy_input,))
    #model = model.to(device)
    #init_weights(model)
# training
    model.train(mode = True)  #must set as that in tranning
    for epoch in range(EPOCH):
        for batch_idx, (data_left,data_right,label) in enumerate(train_loader):
            data_left, data_right = data_left.to(device),data_right.to(device)
            label = label.to(device)
            
            left = model(data_left)
            right = model(data_right)
            
            loss =loss_func.forward(left,right,label)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)], loss: {:.4f}'.format(
                        epoch + 1 , batch_idx * len(data_left), len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader), loss))
		#niter = epoch*len(train_loader)+batch_idx
            	#writer.add_scalar('Train/Loss', loss, niter)
        scheduler.step()
    torch.save(model.state_dict(), "model.pkl")

# Test
    image_test_result = '../image_result/'
    model.eval()
    for data_test,test_image_path in test_loader:
        data_test = data_test.to(device)
        output = model(data_test)
	src = os.path.join(os.path.abspath(vector_test_data_path),test_image_path[0])
        dst = os.path.join(os.path.abspath(image_test_result),str("%.8f" % output.data)+'.csv')
        os.rename(src,dst)
        print(str("%.8f" % output.data),test_image_path)
    
    #writer.close()


if __name__ == "__main__":
    train()
