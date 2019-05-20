import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import os
from PIL import Image
import csv


def get_pairwise_list(path):
    with open(path, 'rb') as myfile:
        reader = csv.reader(myfile)
        return list(reader)
    
class TrainingSets(data.Dataset):
    #each time generate a triplets
    def __init__(self,cvs_file_path,image_folder_path,csv_anchor,image_anchor,transforms=None):
        #is a 2D list contain the index of all pairwise data
        self.pairwise = get_pairwise_list(cvs_file_path)
	self.anchor = get_pairwise_list(csv_anchor)
	self.image_training_folder_path = image_folder_path
	self.image_anchor_path = image_anchor
        if transforms is None:
            self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
        else:
            self.transforms = transforms
        
    def __getitem__(self, index):
        left_path, right_path,label = self.pairwise[index]
        left = Image.open(self.image_training_folder_path + left_path).convert('L')
        right = Image.open(self.image_training_folder_path + right_path).convert('L')
        label = torch.from_numpy(np.array([label], dtype=np.float32))
	anchor_path, anchor_label = self.anchor[np.random.randint(0,len(self.anchor))]
        anchor_image = Image.open(self.image_anchor_path + anchor_path).convert('L')
        anchor_label = torch.from_numpy(np.array([anchor_label], dtype=np.float32))
        return self.transforms(left), self.transforms(right),label,self.transforms(anchor_image),anchor_label

    def __len__(self):
        return len(self.pairwise)

"""    
class TestSets(data.Dataset):
    #each time generate a triplets
    def __init__(self,image_folder_path,transforms=None):
        #is a 2D list contain the index of all images
        
        self.files_name = os.listdir(image_folder_path)
        self.image_test_folder_path = image_folder_path
        if transforms is None:
            self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
        else:
            self.transforms = transforms
            
        
    def __getitem__(self, index):
        #index form 0 to num_test-1
        test_path= self.files_name[index]
        #print(test_path)
        #print(type(test_path))
        test = Image.open(self.image_test_folder_path + str(test_path)).convert('L')
        #test.show(title = test_path )
        return self.transforms(test),test_path

    def __len__(self):
        return len(self.files_name)
"""



class TestSets(data.Dataset):
    #each time generate a triplets
    def __init__(self,csv_file,image_folder_path,transforms=None):
        #is a 2D list contain the index of all images

	self.pairwise = get_pairwise_list(csv_file)
        self.image_test_folder_path = image_folder_path
        if transforms is None:
            self.transforms = T.Compose([
                    T.Resize(256),
                    T.ToTensor(),
                ])
        else:
            self.transforms = transforms


    def __getitem__(self, index):
        #index form 0 to num_test-1
        left_path, right_path,label = self.pairwise[index]
        #print(test_path)
        #print(type(test_path))
	left = Image.open(self.image_test_folder_path + left_path).convert('L')
        right = Image.open(self.image_test_folder_path + right_path).convert('L')
        label = torch.from_numpy(np.array([label], dtype=np.float32))
        #test.show(title = test_path )
        return self.transforms(left), self.transforms(right),label

    def __len__(self):
        return len(self.pairwise)

class CheckSets(data.Dataset):
    #each time generate a triplets
    def __init__(self,image_folder_path,transforms=None):
        #is a 2D list contain the index of all images
        
        self.files_name = os.listdir(image_folder_path)
        self.image_test_folder_path = image_folder_path
        if transforms is None:
            self.transforms = T.Compose([
                    T.Resize(256),
                    T.ToTensor(),
                ])
        else:
            self.transforms = transforms
            
        
    def __getitem__(self, index):
        #index form 0 to num_test-1
        test_path= self.files_name[index]
        #print(test_path)
        #print(type(test_path))
        test = Image.open(self.image_test_folder_path + str(test_path)).convert('L')
        #test.show(title = test_path)
        return self.transforms(test),test_path

    def __len__(self):
        return len(self.files_name)


if __name__ == "__main__":
    image_training_folder_path = '../DataSets/training_set/'
    path_csv_judgment = '../DataSets/result.csv'
    image_test_folder_path ='../DataSets/test_set/'
    image_anchor = '../DataSets/anchor/'
    path_anchor_csv = '../DataSets/anchor.csv'
    a = TrainingSets(path_csv_judgment,image_training_folder_path,path_anchor_csv,image_anchor)
    print(a.__getitem__(32))

