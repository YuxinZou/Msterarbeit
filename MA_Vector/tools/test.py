import torch
import torch.nn as nn
import argparse
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
import time
from tensorboardX import SummaryWriter
sys.path.append("..")
from model.Model_2_ssg import Model
from data.data_loader_process import TrainingSets,TestSets,CheckSets
from tools.loss import CrossEntropy
from tools.FocalLoss import FocalLoss
from tools.utils import init_weights

def train():
    vector_training_data_path = '../DataSets/training_set_preprocess_4096/'
    path_csv_train = '../DataSets/result.csv'
    path_csv_test = '../DataSets/test_set_pairlist_1500_result.csv'
    anchor_file = '../DataSets/anchor/'
    path_anchor_path = '../DataSets/anchor.csv'

    image_test_folder_path ='../DataSets/test_set_preprocess_4096/'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')


#load checkset
    check_set = CheckSets(image_test_folder_path)
    check_loader = torch.utils.data.DataLoader(check_set,batch_size=1,shuffle=False,num_workers=2)
    
#*****************Step2 Setup Model***********************           
    model = Model()
    model.to(device)
    init_weights(model)

#*****************Step3 Setup Optimizer and Loss*********************** 
# loss function
    #loss_funcCE = CrossEntropy()
    loss_funcFL = FocalLoss(gamma = 0)
    loss_MSE = nn.MSELoss()

    pretrained = torch.load("pointnet_ssg_4096_FC_40_sigmoid.pkl")
    model.load_state_dict(pretrained['model_state']) 

    for data_check,check_image_path in check_loader:
        data_check = data_check.to(device)
        output = model.forward(data_check)
        #src = os.path.join(os.path.abspath(image_test_folder_path),test_image_path[0])
        #dst = os.path.join(os.path.abspath(image_test_result),str("%.8f" % output.data)+'.png')
        #os.rename(src,dst)
        print(str("%.8f" % output),check_image_path)
        #print(str("%.8f" % output[1]),check_image_path[1])
        #print(data_check.shape)

if __name__ == "__main__":
    train()
