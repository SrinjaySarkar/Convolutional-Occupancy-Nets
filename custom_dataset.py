import os
import glob
import random
import yaml
from torch.utils.data import Dataset,DataLoader
import torch.utils as utils
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
from PIL import Image

class Field(object):
    def load(self, data_path, idx, category):
        raise NotImplementedError

"""this loads ipoint clouds and points.all 3 of them require transform
data fields is points(randomly sampled points) , the input fields is point clouds and images 
both data and input fields have transforms then fields class and then the loader(get fields function)
flow: (transform(s)=>field=>get field.) """

superpod_root_path="/lustre/scratch/client/vinai/users/sarkar/conv_occ/ShapeNetCore.v2/"
root_file_path="/vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/02691156/"
# val_ids=[]
# with open(os.path.join(root_file_path,"val.lst")) as f:
#     linelist=f.readlines()
#     val_ids.append(linelist)

#transform
class subsample_points(object):
    def __init__(self,N):
        self.N=N
    
    def __call__(self,data):
        points=data[None]
        occ=data["occ"]
        data_out=data.copy()
        if isinstance(self.N,int):
            idx=np.random.randint(points.shape[0],size=self.N)
            data_out.update({None:points[idx,:],"occ":occ[idx]})
        return (data_out)
#field
class points_field(Field):
    def __init__(self,file_name,transform=None,with_transforms=False,unpackbits=False):
        self.file_name=file_name
        self.transform=transform
        self.with_transforms=with_transforms
        self.unpackbits=unpackbits
    
    #this function is loading the points file(points.npz)(contains both off and on surface points)
    #chnage it to load point cloud files instead (pointcloud.npz)
    def load(self,model_path,idx,category):
        file_path=os.path.join(model_path,self.file_name)
        points_dict=np.load(file_path)
        # print(points_dict)
        
        points=points_dict["points"]
        if points.dtype == np.float16:
            points=points.astype(np.float32)
            points+=1e-4 * np.random.randn(*points.shape)
        else:
            points=points.astype(np.float32)
        
        occupancies=points_dict["occupancies"]
        occupancies=np.unpackbits(occupancies)[:points.shape[0]]
        occupancies=occupancies.astype(np.float32)
        
        data={None:points,"occ":occupancies,}
        if self.transform is not None:
            data=self.transform(data)
        return (data)



#get field
def get_data_fields(mode,file="tr"):
    points_transform=subsample_points(2048)
    with_transform=False

    fields={}
    if file == "tr":
        point_file="points_final.npz"#tr,t,r
    elif file == "r":
        point_file="points_t1.npz"
    else:
        point_file="points.npz"
    fields["points"]=points_field(point_file,points_transform,with_transforms=False,unpackbits=True)

    # if mode in ("val","test"):
    #     points_iou_file="points.npz"
    #     fields["points_iou"]=points_field(points_iou_file,trsnforms=None,with_transforms=False,unpakckbits=true)
    # if mode in ("val","test"):
    #     points_iou_file="points.npz"
    #     fields["points_iou"]=points_field(points_iou_file,with_transforms=False,unpackbits=True)
    
    return (fields)

#transform
class pointcloud_noise(object):
    def __init__(self,dev):
        self.dev=dev
    
    def __call__(self,data):
        data_out=data.copy()
        points=data[None]
        noise=self.dev * np.random.randn(*points.shape)
        noise=noise.astype(np.float32)
        assert(points.shape == noise.shape)
        data_out[None]=points+noise
        return (data_out)
#transform
class subsample_pointcloud(object):
    def __init__(self,N):
        self.N=N
    
    def __call__(self,data):
        data_out=data.copy()
        points=data[None]
        normals=data["normals"]
        indices=np.random.randint(points.shape[0],size=self.N)
        data_out[None]=points[indices,:]
        data_out["normals"]=normals[indices,:]
        return (data_out)
#field
class point_cloud_field(Field):
    def __init__(self,file_name,transform=None,with_transforms=False):
        self.file_name=file_name
        self.transform=transform
        self.with_transforms=with_transforms
    
    def load(self,model_path,idx,category):
        file_path=os.path.join(model_path,self.file_name)
        pointcloud_dict=np.load(file_path)
        points=pointcloud_dict["points"].astype(np.float32)
        normals=pointcloud_dict["normals"].astype(np.float32)

        data={None:points,"normals":normals}
        
        if self.transform is not None:
            # print(self.transform)
            data=self.transform(data)
            # print(data)
        #print(data)
        return (data)



#inputs field
def get_input_fields(mode,type="pointcloud",file="tr"):
    #here I am using the same inputs field to fetch both images and point clouds. 
    assert type in ("pointcloud","image","all")
    input_type="pointcloud"
    with_transforms=False
    transform=transforms.Compose([pointcloud_noise(0.005),subsample_pointcloud(3000)])
    if file == "tr":
        pointcloud_folder="pointcloud_final.npz"#tr,r,t
    elif file == "r":
        pointcloud_folder="pointcloud_t1.npz"
    else : 
        pointcloud_folder="pointcloud.npz"
    inputs_field_pointcloud=point_cloud_field(file_name=pointcloud_folder,with_transforms=False,transform=transform)
    
    if type == "pointcloud":
        return (inputs_field_pointcloud)
    else:
        return (None)


class Shapes3dDataset(utils.data.Dataset):  
    def __init__(self,dataset_folder,fields,categories=None,split=None,transform=None):
        self.dataset_folder=dataset_folder
        self.fields=fields
        self.transform=transform

        if categories is None:
            categories=os.listdir(dataset_folder)
            categories=[c for c in categories if os.path.isdir(os.path.join(dataset_folder,c))]
        # print(categories)
        
        metadata_file=os.path.join(dataset_folder,"metadata.yaml")
        if os.path.exists(metadata_file):
            with open(metadata_file,"r") as f:
                self.metadata=yaml.load(f)
        else:
            self.metadata={c:{"id":c,"name":"n/a"} for c in categories}
        for c_idx,c in enumerate(categories):
            self.metadata[c]["idx"]=c_idx
        # print(self.metadata)
        
        self.models=[]
        # print("cats",categories)
        for c_idx,c in enumerate(categories):
            subpath=os.path.join(dataset_folder,c)
            if not (os.path.exists(subpath)):
                print("FIX PATH",subpath)
                break   
            split_file=os.path.join(subpath,split+".lst")
            with open(split_file,"r") as f:
                models_c=f.read().split("\n")
            
            self.models+=[{"category":c,"model":m} for m in models_c]
    
    def __len__(self):
        length=len(self.models)
        return (length)

    def __getitem__(self,idx):
        category=self.models[idx]["category"]
        model=self.models[idx]["model"]
        c_idx=self.metadata[category]["idx"]

        if "ShapeNetCore.v2" or "ShapeNetCore.v2" in self.dataset_folder.split("/"):
            model_path=os.path.join(self.dataset_folder,category,model,"models")
        else:
            model_path=os.path.join(self.dataset_folder,category,model)
        data={}
        for field_name,field in self.fields.items():
            # print(field_name)
            # print(field)
            field_data=field.load(model_path,idx,c_idx)
            # print(field_data)
            try:
                field_data=field.load(model_path,idx,c_idx)
                # print("ok")
                # print(field_data)
            except Exception : 
                print("Error for loading field %s of model %s"%(field_name,model))
                return None
        
            if isinstance (field_data,dict):
                for k,v in field_data.items():
                    if k is None:
                        data[field_name]=v
                    else:
                        data["%s.%s"%(field_name,k)]=v
            else:
                data[field_name]=field_data
        
        if self.transform is not None:
            data=self.transform(data)
        data["category"]=category
        return (data)
    
    def get_model_dict(self,idx):
        return (self.models[idx])


# data_loader=Shapes3dDataset(dataset_folder,fields,split=split,categories=categories)

#change dataset_folder on line 245 to /vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/ and also add"models" on line 200 for shapenet v2
#change dataset folder on line 245 /vinai-public-dataset/Shapenet_DATASET/ShapeNet/ and remove "models" on line 200

def get_dataset(mode,file,categories,dataset_folder,return_idx=False,return_category=False):
    method="onet"
    dataset_type="Shapes3D"
    # dataset_folder="/vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/"

    splits={"train":"train","val":"val","test":"test"}
    split=splits[mode]
    
    file=file
    if dataset_folder == "/vinai-public-dataset/Shapenet_DATASET/ShapeNet/":
        file=None
    fields=get_data_fields(mode,file=file)#tr,t,r,none
    inputs_field_pc=get_input_fields(mode,type="pointcloud",file=file)#tr,t,r,none

    assert (inputs_field_pc is not None)
    fields["pointcloud"]=inputs_field_pc
    dataset=Shapes3dDataset(dataset_folder,fields,split=split,categories=categories)
    return (dataset)

# v2_path="/vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/"
# convocc_path="/vinai-public-dataset/Shapenet_DATASET/ShapeNet/"

# f1=get_dataset(mode="val",file="tr",categories=["02691156"],dataset_folder=v2_path)
# print(len(f1))
# print(f1[0].keys())
# print(f1[0]["points"].shape)
# print(f1[0]["points.occ"].shape)
# print(f1[0]["pointcloud"].shape)
# print(f1[0]["category"])

# f2=get_dataset(mode="val",file="tr",categories=["02691156"],dataset_folder=convocc_path)
# print(len(f2))
# id=0
# print(f2[id].keys())
# print(f2[id]["points"].shape)
# print(f2[id]["points.occ"].shape)
# print(f2[id]["pointcloud"].shape)
# print(f2[id]["category"])



# import random
# idx=random.randint(0,len(f))
# print(np.max(f[idx]["points"]))#(default:(-0.55.0.55))(v2:(-0.55,0.55))
# print(np.min(f[idx]["points"]))
# print(np.max(f[idx]["pointcloud"]))#(default:(-0.5.0.5))(v2:())
# print(np.min(f[idx]["pointcloud"]))





# def normalize_3d_coordinate(p, padding=0.1):
#     ''' Normalize coordinate to [0, 1] for unit cube experiments.
#         Corresponds to our 3D model

#     Args:
#         p (tensor): point
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.55, 0.55] -> [-0.5, 0.5]
#     '''
    
#     p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
#     p_nor = p_nor + 0.5 # range (0, 1)
#     # f there are outliers out of the range
#     # if p_nor.max() >= 1:
#     #     p_nor[p_nor >= 1] = 1 - 10e-4
#     # if p_nor.min() < 0:
#     #     p_nor[p_nor < 0] = 0.0
#     return p_nor


# points=f[idx]["points"]
# pointcloud=f[idx]["pointcloud"]

# print("##Before##")
# print(np.min(points))
# print(np.max(points))
# print(np.min(pointcloud))
# print(np.max(pointcloud))

# points=normalize_3d_coordinate(points,padding=0.1)
# pointcloud=normalize_3d_coordinate(pointcloud,padding=0.1)

# print("##After##")
# print(np.min(points))
# print(np.max(points))
# print(np.min(pointcloud))
# print(np.max(pointcloud))


# #finding optimum value for padding
# points=f[idx]["points"]
# pointcloud=f[idx]["pointcloud"]
# print("optimum")
# padding=0.0
# p_nor=pointcloud/(1 + padding + 10e-4)#[-0.5,0.5]#[-0.41,0.41]
# p_nor=p_nor + 0.4
# print(np.min(p_nor))
# print(np.max(p_nor))