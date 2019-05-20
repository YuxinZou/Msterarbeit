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
from model.Model import Model
from data.data_loader import TrainingSets,TestSets,CheckSets
from tools.loss import CrossEntropy
from tools.FocalLoss import FocalLoss
from tools.utils import init_weights

def validation():

    image_training_folder_path = '../DataSets/training_set/'
    path_csv_judgment = '../DataSets/result.csv'
    path_csv_test = '../DataSets/test_set_pairlist_1500_result.csv'
    image_test_folder_path ='../DataSets/test_set/'
    image_anchor = '../DataSets/anchor/'
    path_anchor_csv = '../DataSets/anchor.csv'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#*****************Step1 Setup Data***********************
#load validationset
    test_set = TestSets(path_csv_test,image_training_folder_path)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=25,shuffle=False,num_workers=2)
    
#*****************Step2 Setup Model***********************           
    model = Model()
    model.to(device)
    pretrained = torch.load("resnet_attention_50_CE_with_sigmoid.pkl")
    model.load_state_dict(pretrained['model_state']) 
    model.eval()
    print('sucess till now')

#*****************Step3 Setup Optimizer and Loss*********************** 
# loss function
    #loss_funcCE = CrossEntropy()
    loss_funcFL = FocalLoss(gamma = 0)
    loss_MSE = nn.MSELoss()

    TP,TN,FN,FP,acc,F1 = 0,0,0,0,0,0
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0

        for data_left,data_right,label in test_loader:
	    data_left, data_right = data_left.to(device),data_right.to(device)
            label = label.to(device)
            left = model(data_left)
            right = model(data_right)
	    _, val_predic =loss_funcFL.forward(left,right,label)
	    val_correct +=(val_predic == label.long()).sum().float()
	    val_total += len(label)
            #print(val_predic,label)
            #print(val_correct)
	    TP += ((val_predic == 1) & (label.long() == 1)).sum().float()
            #print(TP)
	    TN += ((val_predic == -1) & (label.long() == -1)).sum().float()
	    FN += ((val_predic == -1) & (label.long() == 1)).sum().float()
	    FP += ((val_predic == 1) & (label.long() == -1)).sum().float()
            #print(TN,FN,FP)
	test_accuracy = val_correct/val_total
	p = TP / (TP+FP)
	r = TP / (TP + FN)
	F1 = 2 * r *p / (r + p)
	acc = (TP + TN) / (TP + TN + FP + FN)
        print('Validation Accuracy: {:.4f}%'.format(100 * test_accuracy))
	print('Precision: {:.4f}%'.format(100 * p))
	print('Recall: {:.4f}%'.format(100 * r))
	print('F1: {:.4f}%'.format(100 * F1))
	print('acc: {:.4f}%'.format(100 * acc))

	

if __name__ == "__main__":
    validation()
