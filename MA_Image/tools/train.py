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

def train(args):

    image_training_folder_path = '../DataSets/training_set/'
    path_csv_judgment = '../DataSets/result.csv'
    path_csv_test = '../DataSets/test_set_pairlist_1500_result.csv'
    image_test_folder_path ='../DataSets/test_set/'
    image_anchor = '../DataSets/anchor/'
    path_anchor_csv = '../DataSets/anchor.csv'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')


#*****************Step1 Setup Data***********************
#load trainingset
    train_set = TrainingSets(path_csv_judgment,image_training_folder_path,path_anchor_csv,image_anchor) 
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=2)
#load testset
    test_set = TestSets(path_csv_test,image_training_folder_path)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=2)
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

# optimizer
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, momentum=0.9)  
    #EPOCH = 40
    #optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,betas=(0.9,0.999))  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
#Resume Model
    start_epoch = 0
    best_acc = -100
    if args.resume:
	if os.path.isfile(args.resume):
            print("> Loading model and optimizer from checkpoint '{}'".format(args.resume))
	    checkpoint = torch.load(args.resume)
	    start_epoch = checkpoint['epoch']
	    best_acc = checkpoint['best_acc']
    	    model.load_state_dict(checkpoint['model_state']) 
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("> Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))

        else:
            print("> No checkpoint found at '{}'".format(args.resume))
            raise Exception("> No checkpoint found at '{}'".format(args.resume))


# set tensor-board for visualization
    writer = SummaryWriter('runs/'+args.log_root)

# training
    start_time = time.time()
    for epoch in range(start_epoch,args.num_epochs):
	model.train(mode = True)  #must set as that in tranning
	train_correct = 0
        train_total = 0
        for batch_idx, (data_left,data_right,label,anchor_image,label_anchor) in enumerate(train_loader):
            data_left, data_right,anchor_image = data_left.to(device),data_right.to(device),anchor_image.to(device)
            label = label.to(device)
            label_anchor = label_anchor.to(device)
            #print(label_anchor)
            left = model(data_left)
            right = model(data_right)
            anchor = model(anchor_image)
            
            #loss =loss_func.forward(left,right,label) + 0.5 * loss_MSE(anchor,label_anchor)
	    CE_loss, train_predic =loss_funcFL.forward(left,right,label)
            MSE_loss = loss_MSE(anchor,label_anchor)
            train_loss = CE_loss  + 0.5 * MSE_loss
            optimizer.zero_grad()
            train_loss.backward()
	    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            optimizer.step()
	    # calculate the accuracy
	    train_correct +=(train_predic == label.long()).sum().float()
	    train_total += len(label)

            if batch_idx % 5 == 0:
		#print the loss
                print('Train Epoch: {} [{}/{} ({:.0f}%)], CE_loss: {:.4f},MSE_loss: {:.4f},tain_loss: {:.4f}'.format(
                        epoch + 1 , batch_idx * len(data_left), len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader), CE_loss, MSE_loss, train_loss))

		if args.tensorboard:
		    niter = epoch*len(train_loader)+batch_idx
            	    writer.add_scalar('Train/Loss', train_loss, niter)
                    writer.add_scalar('CE/Loss', CE_loss, niter)
                    writer.add_scalar('MSE/Loss', MSE_loss, niter)

		
	train_accuracy =  train_correct/train_total
	if args.tensorboard:
            writer.add_scalar('Train/Accuracy', train_accuracy, epoch+1)
	print('Train Epoch: {} ,Train Accuracy: {:.4f}%'.format(epoch + 1 , 100 * train_accuracy))
	
    
# Validation
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
	test_accuracy = val_correct/val_total
        print('Train Epoch: {} ,Validation Accuracy: {:.4f}%'.format(epoch + 1 ,100 * test_accuracy))
	if args.tensorboard:
            writer.add_scalar('Test/Accuracy', test_accuracy, epoch+1)
	
	if test_accuracy >= best_acc:
            best_acc = test_accuracy
            state = {"epoch": epoch + 1,
                     "best_acc": best_acc,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict()}
            torch.save(state, args.model_para)

        # Note that step should be called after validate()
        scheduler.step()
    
    if args.tensorboard:
    	writer.close()
    print('best_acc: {}'.format(best_acc))
    print("Training time: {}s".format(time.time() - start_time))
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Training Done!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


#Test
    #image_test_result = '/zfs/zou/project/image_result/'
    model.eval()
    for data_check,check_image_path in check_loader:
        data_check = data_check.to(device)
        output = model(data_check)
	#src = os.path.join(os.path.abspath(image_test_folder_path),test_image_path[0])
        #dst = os.path.join(os.path.abspath(image_test_result),str("%.8f" % output.data)+'.png')
        #os.rename(src,dst)
        print(str("%.8f" % output.data),check_image_path)

"""
    pretrained = torch.load("resnet18_30_CE_with_MSE.pkl")
    model.load_state_dict(pretrained['model_state'])
    model.eval()
    for data_check,check_image_path in check_loader:
        data_check = data_check.to(device)
        output = model(data_check)
        #src = os.path.join(os.path.abspath(image_test_folder_path),test_image_path[0])
        #dst = os.path.join(os.path.abspath(image_test_result),str("%.8f" % output.data)+'.png')
        #os.rename(src,dst)
        print(str("%.8f" % output.data),check_image_path)
"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HyperParameters')

    parser.add_argument('--num_epochs', nargs='?', type=int, default = 50,
                        help='number of the epochs used for training process | 40 by default')
    parser.add_argument('--batch_size', nargs='?',type=int, default=36, 
                        help='Batch size | 64 by default')
    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.01,
                        help='Learning rate | 0.005 by default')

    parser.add_argument('--resume', nargs='?', type=str, default=None,  
                        help='Path to previous saved model to restart from | None by  default')
    parser.add_argument('--pre_trained', nargs='?', type=str, default=None,
                        help='Path to pre-trained  model to init from | None by default')

    parser.add_argument('--tensorboard', nargs='?', type=bool, default=True,
                        help='visualization using tensorboard | True by default')

    parser.add_argument('--log_root', nargs='?', type=str, default='expresnet_attention__50_CE_with_sigmoid',
                        help='Folder to store tensorboard | expresnet34_40 by default')

    parser.add_argument('--model_para', nargs='?', type=str, default='resnet_attention_50_CE_with_sigmoid.pkl',
                        help='best saved model parameter | resnet34_40.pkl by default')
    train_args = parser.parse_args()

    train(train_args)
