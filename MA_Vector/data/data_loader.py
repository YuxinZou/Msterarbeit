import os
import random
import numpy as np
import torch
import torch.utils.data as data
import csv


def scale_linear_bycolumn(rawdata, high=1.0, low=0.0):
    mins = np.min(rawdata, axis=0)
    maxs = np.max(rawdata, axis=0)
    rng = maxs - mins
    return high - (high-low)*(maxs-rawdata)/(rng+np.finfo(np.float32).eps)

def get_pairwise_list(path):
    with open(path, 'rb') as myfile:
        reader = csv.reader(myfile)
        return list(reader)

class TrainingSets(data.Dataset):
    def __init__(self, csv_file_path,vector_training_data_path,npoints=2048):
        self.csv_file_path = csv_file_path
        self.vector_training_data_path = vector_training_data_path
        self.npoints = npoints
	self.pairwise = get_pairwise_list(self.csv_file_path)

    def __getitem__(self, index):
	left_path, right_path,label = self.pairwise[index]
        points_left = np.loadtxt(self.vector_training_data_path + left_path,delimiter=',',dtype=np.float32)
        points_right = np.loadtxt(self.vector_training_data_path + right_path,delimiter=',',dtype=np.float32)
        replace_left = True if points_left.shape[0]<self.npoints else False
	replace_right = True if points_right.shape[0]<self.npoints else False
        #make sure the number of point is same
        choice_left = np.random.choice(points_left.shape[0], self.npoints, replace=replace_left)
        choice_right = np.random.choice(points_right.shape[0], self.npoints, replace=replace_right)
        points_left = points_left[choice_left, :]
        points_right = points_right[choice_right, :]
        points_left = scale_linear_bycolumn(points_left)
        points_right = scale_linear_bycolumn(points_right)
        points_left = torch.from_numpy(points_left).type(torch.FloatTensor)
	#print(points_left,points_left.shape)
	points_left = torch.transpose(points_left,0,1)
	#print(points_left,points_left.shape)
        points_right = torch.from_numpy(points_right).type(torch.FloatTensor)
	points_right = torch.transpose(points_right,0,1)
        label = torch.from_numpy(np.array([label], dtype=np.float32))
        return points_left, points_right, label

    def __len__(self):
        return len(self.pairwise)

"""
class TestSets(data.Dataset):
        def __init__(self,vector_test_data_path,npoints=2048):
	    self.vector_test_data_path = vector_test_data_path
            self.npoints = npoints
	    self.files_name = os.listdir(self.vector_test_data_path)

        def __getitem__(self, index):
	    test_path= self.files_name[index]
            points_test = np.loadtxt(self.vector_test_data_path + test_path ,delimiter=',',dtype=np.float32)
            replace = True if points_test.shape[0]<self.npoints else False
	    #make sure the number of point is same
            choice = np.random.choice(points_test.shape[0], self.npoints, replace=replace)
            points_test = points_test[choice, :]
            points_test = scale_linear_bycolumn(points_test)
            points_test = torch.from_numpy(points_test).type(torch.FloatTensor)
	    points_test = torch.transpose(points_test,0,1)
            return points_test,test_path

        def __len__(self):
            return len(self.files_name)
"""
class TestSets(data.Dataset):
        def __init__(self, csv_file_path,vector_training_data_path,npoints=2048):
            self.csv_file_path = csv_file_path
	    self.vector_training_data_path = vector_training_data_path
            self.npoints = npoints
	    self.pairwise = get_pairwise_list(self.csv_file_path)

        def __getitem__(self, index):
	    left_path, right_path,label = self.pairwise[index]
            points_left = np.loadtxt(self.vector_training_data_path + left_path,delimiter=',',dtype=np.float32)
            points_right = np.loadtxt(self.vector_training_data_path + right_path,delimiter=',',dtype=np.float32)
            replace_left = True if points_left.shape[0]<self.npoints else False
	    replace_right = True if points_right.shape[0]<self.npoints else False
            #make sure the number of point is same
            choice_left = np.random.choice(points_left.shape[0], self.npoints, replace=replace_left)
            choice_right = np.random.choice(points_right.shape[0], self.npoints, replace=replace_right)
            points_left = points_left[choice_left, :]
            points_right = points_right[choice_right, :]
            points_left = scale_linear_bycolumn(points_left)
            points_right = scale_linear_bycolumn(points_right)
            points_left = torch.from_numpy(points_left).type(torch.FloatTensor)
	    #print(points_left,points_left.shape)
	    points_left = torch.transpose(points_left,0,1)
	    #print(points_left,points_left.shape)
            points_right = torch.from_numpy(points_right).type(torch.FloatTensor)
	    points_right = torch.transpose(points_right,0,1)
            label = torch.from_numpy(np.array([label], dtype=np.float32))
            return points_left, points_right, label

        def __len__(self):
            return len(self.pairwise)



if __name__ == '__main__':
    vector_training_data_path = '../DataSets/training_set/'
    csv_file_path = '../DataSets/result.csv'
    #vector_test_data_path ='../test_set/'
    a = TrainingSets(csv_file_path,vector_training_data_path)
    print(a.__getitem__(2))
