import os
import random
import numpy as np
import torch
import torch.utils.data as data
import csv

def get_pairwise_list(path):
    with open(path, 'rb') as myfile:
        reader = csv.reader(myfile)
        return list(reader)

class TrainingSets(data.Dataset):
    def __init__(self, csv_file_path,vector_training_data_path,path_anchor,path_csv_anchor,npoints=2048):
        self.csv_file_path = csv_file_path
	self.vector_training_data_path = vector_training_data_path
	self.path_anchor = path_anchor
        self.anchor = get_pairwise_list(path_csv_anchor)
        self.npoints = npoints
	self.pairwise = get_pairwise_list(self.csv_file_path)

    def __getitem__(self, index):
	left_path, right_path,label = self.pairwise[index]
        points_left = np.loadtxt(self.vector_training_data_path + left_path,delimiter=',',dtype=np.float32)
        points_right = np.loadtxt(self.vector_training_data_path + right_path,delimiter=',',dtype=np.float32)
        points_left = torch.from_numpy(points_left).type(torch.FloatTensor)
        points_right = torch.from_numpy(points_right).type(torch.FloatTensor)
        label = torch.from_numpy(np.array([label], dtype=np.float32))

	anchor_path, anchor_label = self.anchor[np.random.randint(0,len(self.anchor))]
	anchor_vector = np.loadtxt(self.path_anchor + anchor_path,delimiter=',',dtype=np.float32)
        anchor_label = torch.from_numpy(np.array([anchor_label], dtype=np.float32))
        return points_left, points_right, label,anchor_vector,anchor_label

    def __len__(self):
        return len(self.pairwise)


class TestSets(data.Dataset):
    def __init__(self,csv_file_path,vector_test_data_path,npoints=2048):
        self.csv_file_path = csv_file_path
        self.vector_test_data_path = vector_test_data_path
        self.npoints = npoints
	self.pairwise = get_pairwise_list(self.csv_file_path)

    def __getitem__(self, index):
        left_path, right_path,label = self.pairwise[index]
        points_left = np.loadtxt(self.vector_test_data_path + left_path,delimiter=',',dtype=np.float32)
        points_right = np.loadtxt(self.vector_test_data_path + right_path,delimiter=',',dtype=np.float32)
        points_left = torch.from_numpy(points_left).type(torch.FloatTensor)
        #print(points_left,points_left.shape)
        points_right = torch.from_numpy(points_right).type(torch.FloatTensor)
        label = torch.from_numpy(np.array([label], dtype=np.float32))
        return points_left, points_right, label

    def __len__(self):
        return len(self.pairwise)

class CheckSets(data.Dataset):
    #each time generate a triplets
    def __init__(self,vecotr_folder_path,transforms=None):
        #is a 2D list contain the index of all images

        self.files_name = os.listdir(vecotr_folder_path)
        self.vector_test_folder_path = vecotr_folder_path


    def __getitem__(self, index):
        #index form 0 to num_test-1
        test_path= self.files_name[index]
	test = np.loadtxt(self.vector_test_folder_path + str(test_path),delimiter=',',dtype=np.float32)
        test = torch.from_numpy(test).type(torch.FloatTensor)
        return test,test_path

    def __len__(self):
        return len(self.files_name)


if __name__ == '__main__':
    vector_training_data_path = '../DataSets/training_set_preprocess_2048/'
    csv_file_path = '../DataSets/result_vector_with_0_50%.csv'
    vector_test_data_path ='../DataSets/test_set_preprocess_2048/'
    anchor_file = '../DataSets/anchor/'
    anchor_csv ='../DataSets/anchor.csv'
    a = TrainingSets(csv_file_path,vector_training_data_path,anchor_file,anchor_csv)
    print(a.__getitem__(2))
