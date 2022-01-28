import os
import torch
import json
import random
import h5py
from tqdm import tqdm
from glob import glob
import numpy as np
import torch.utils as utils 
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def translate_pointcloud(pointcloud):
    xyz1=np.random.uniform(low=2./3., high=3./2.,size=[3])
    xyz2=np.random.uniform(low=-0.2, high=0.2,size=[3])
    translated_pointcloud=np.add(np.multiply(pointcloud, xyz1),xyz2).astype('float32')
    return (translated_pointcloud)

def jitter_pointcloud(pointcloud,sigma=0.01,clip=0.02):
    N,C=pointcloud.shape
    pointcloud+=np.clip(sigma*np.random.randn(N, C),-1*clip, clip)
    return (pointcloud)

def rotate_pointcloud(pointcloud):
    theta = np.pi*2*np.random.choice(24) / 24
    rotation_matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    pointcloud[:,[0,2]]=pointcloud[:,[0,2]].dot(rotation_matrix)#random rotation (x,z)
    return (pointcloud)


class recon_dataset(utils.data.Dataset):
    def __init__(self,root,dataset_name="shapenetcorev2",num_points=2048,split="train",class_choice=None,load_name=False,load_file=False,random_rotate=False,
    random_jitter=False,random_translate=False):
                assert dataset_name.lower() in ["shapenetcorev2","modelnet40"]
                assert num_points <= 2048

                if dataset_name in ["shapenetpart","shapenetcorev2"]:
                    assert split.lower() in ["train","test","val","trainval","all"]
                else:
                    assert split.lower() in ["train","test","all"]
                self.root=os.path.join(root,dataset_name+"_hdf5_2048")
                self.dataset_name=dataset_name
                self.num_points=num_points
                self.split=split
                self.load_name=load_name
                self.class_choice=class_choice
                self.load_file=load_file
                self.random_rotate=random_rotate
                self.random_jitter=random_jitter
                self.random_translate=random_translate
                self.path_h5py_all = []
                self.path_name_all = []
                self.path_json_all = []
                self.path_file_all = []
                if self.split in ['train','trainval','all']:   
                    self.get_path('train')
                    # print(self.path_h5py_all)
                if self.dataset_name in ['shapenetpart','shapenetcorev2']:
                    if self.split in ['val','trainval','all']: 
                        self.get_path('val')
                if self.split in ['test', 'all']:   
                    self.get_path('test')

                self.path_h5py_all.sort()
                # print("$$$$",self.path_h5py_all)
                data, label=self.load_h5py(self.path_h5py_all)
                if self.load_name or self.class_choice!=None:
                    self.path_name_all.sort()
                    self.name=self.load_json(self.path_json_all)    # load label name
                if self.load_file:
                    self.path_file_all.sort()
                    self.file = self.load_json(self.path_file_all)
                self.data=np.concatenate(data,axis=0)
                self.label=np.concatenate(label,axis=0)

                if self.class_choice!=None:
                    indices=(self.name == class_choice)
                    print(indices)
                    self.data=self.data[indices]
                    self.label=self.label[indices]
                    if self.load_file:
                        self.file=self.file[indices]
    
    def get_path(self,type):
        path_h5py=os.path.join(self.root,'%s*.h5'%type)
        # print("Here",path_h5py)
        self.path_h5py_all+=glob(path_h5py)
        # print("here",self.path_h5py_all)
        if self.load_name:
            path_json=os.path.join(self.root,'%s*_id2name.json'%type)
            self.path_json_all+=glob(path_json)

    def load_h5py(self,path):
        all_data=[]
        all_label=[]
        for h5_name in path:
            # print("$$$$$",h5_name)
            f=h5py.File(h5_name,"r")
            data=f["data"][:].astype("float32")
            label=f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        return (all_data,all_label)
    
    def load_json(self,path):
        all_data=[]
        for json_name in path:
            j=open(json_name,"r+")
            data=json.load(j)
            all_data+=data
        return (all_data)
    
    def __len__(self):
        size=self.data.shape[0]
        return (size)
    
    def __getitem__(self,item):
        point_set=self.data[item][:self.num_points]
        label=self.label[item]
        if self.load_name :
            name=self.name[item]
        if self.random_rotate:
            point_set=rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set=jitter_pointcloud(point_set)
        if self.random_translate:
            point_set=translate_pointcloud(point_set)
        
        point_set=torch.from_numpy(point_set)
        label=torch.from_numpy(np.array([label]).astype(np.int64))
        label=label.unsqueeze(0)
        if self.load_name:
            return(point_set,label,name)
        else:
            return(point_set,label)
root_path="/vinai/sskar/TTA"
TEST_DATASET=recon_dataset(root=root_path,dataset_name="shapenetcorev2",num_points=2048,split='val',class_choice=None,
load_name=False,random_rotate=False,random_jitter=False,random_translate=False)
# testDataLoader=torch.utils.data.DataLoader(TEST_DATASET,batch_size=bs,shuffle=True,num_workers=10,drop_last=False)
print(TEST_DATASET[0][0].shape)
print(TEST_DATASET[0][1])