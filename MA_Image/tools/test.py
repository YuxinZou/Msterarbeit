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
from tensorboardX import SummaryWriter
sys.path.append("..")
from model.Model import Model
from data.data_loader import TrainingSets,TestSets,CheckSets
from tools.loss import CrossEntropy
from tools.FocalLoss import FocalLoss
from tools.utils import init_weights

def test_image(args):
    image_training_folder_path = '../DataSets/training_set/'
    path_csv_test = '../DataSets/test_set_pairlist_1500_result.csv'
    path_csv_result = '../DataSets/wrong_classified_data.csv'
    image_test_folder_path ='../DataSets/test_set/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = TestSets(path_csv_test,image_training_folder_path)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=2)
    

    check_set = CheckSets(image_test_folder_path)
    check_loader = torch.utils.data.DataLoader(check_set,batch_size=1,shuffle=False,num_workers=2)    
    #set model and load parameter        
    model = Model()
    model.to(device)
    pretrained = torch.load("resnet18_50_CE_with_MSE_25_sigmoid.pkl")
    model.load_state_dict(pretrained['model_state']) 
    model.eval()
    loss_funcFL = FocalLoss(gamma = 0)
    print('sucess till now')



    for data_check,check_image_path in check_loader:
        data_check = data_check.to(device)
        output = model(data_check)
        #output = F.sigmoid(output*2-1)
        #src = os.path.join(os.path.abspath(image_test_folder_path),test_image_path[0])
        #dst = os.path.join(os.path.abspath(image_test_result),str("%.8f" % output.data)+'.png')
        #os.rename(src,dst)
        print(str("%.8f" % output.data),check_image_path)

"""    
    with torch.no_grad():  
        for data_left,data_right,label,left_path,right_path in test_loader:
	    data_left, data_right = data_left.to(device),data_right.to(device)
            label = label.to(device)
            left = model(data_left)
            right = model(data_right)
	    _, predic =loss_funcFL.forward(left,right,label)
            #print(predic)

 	    index = np.where(predic.cpu().numpy() != label.cpu().long().numpy())[0]
            index = np.array(index)
	    print(index)
            for idx in index:
	        with open(path_csv_result,'ab') as file_write:
            	    writer = csv.writer(file_write)
           	    writer.writerow([left_path[idx],right_path[idx]])		
"""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument('--type', nargs='?', type=str, default='image',
                        help='type of input, image or vector')
    parser.add_argument('--num_epochs', nargs='?', type=int, default=40,
                        help='number of the epochs used for training process | 40 by default')
    parser.add_argument('--batch_size', nargs='?',type=int, default=30, 
                        help='Batch size | 64 by  default')
    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.005,
                        help='Learning rate | 0.005 by  default')

    parser.add_argument('--resume', nargs='?', type=str, default=None,  
                        help='Path to previous saved model to restart from | None by  default')
    parser.add_argument('--pre_trained', nargs='?', type=str, default=None,
                        help='Path to pre-trained  model to init from | None by  default')

    parser.add_argument('--tensorboard', nargs='?', type=bool, default=True,
                        help='visualization using tensorboard | True by  default')
    test_args = parser.parse_args()
    test_image(test_args)
