#find all ids for which mesh existed 
#then do mise

#reconstruction imports
##reconstruction imports


import argparse
import collections
import trimesh
import itertools
import numpy as np
import pandas as pd
import os
import json
import re
import glob
import sys
import open3d as o3d
from multiprocessing import Pool
from functools import partial

import sys
sys.path.append('/vinai/sskar/cvpr22/')
# from dataset import get_dataset
from model import reconstruction,ReconstructionNet
from loss import emdModule, ChamferLoss


import torch
import os
import numpy as np
import shutil
import random
from torch.nn import functional as F
from torch import distributions as dist
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
from collections import OrderedDict
import pandas as pd
from torch import distributions as dist
from src import config
from src.utils import libmcubes
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
from src.utils.voxels import VoxelGrid


##
from src.conv_onet import models, training
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
from src.utils.voxels import VoxelGrid
from src.encoder import encoder_dict
from src.conv_onet import generation

##
from v2_datalaoder import recon_dataset
import trimesh
from custom_dataset import get_dataset
from src.utils import libmcubes
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""bag(X),cap,car,chair,Earphone,Guitar,Knife,Lamp,Laptop,Motor,Mug,Pistol,Rocket,Skate,Table"""

catfile="/vinai/sskar/TTA/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt"
cat={}
normal_channel=False#normal_channel
with open(catfile,"r") as f:
    for line in f :
        ls=line.strip().split()
        cat[ls[0]]=ls[1]
cat={k:v for k,v in cat.items()}


root='/vinai/sskar/TTA/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
v1_path="/vinai-public-dataset/shapenet_corev1/ShapeNetCore.v1/"
v2_path="/vinai-public-dataset/shapenet_corev2/ShapeNetCore.v2/"
##########################################################################################################################
class_choice="Cap"

points_size=100000
pointcloud_size=100000
points_uniform_ratio=0.3
points_padding=0.1
points_sigma=0.01
if not class_choice is  None:
    cat = {k:v for k,v in cat.items() if k in class_choice}
meta={}
with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
    train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
with open(os.path.join(root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
    val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
with open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
    test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

for item in cat:
    # print('category', item)
    meta[item]=[]
    dir_point=os.path.join(root,cat[item])
    fns=sorted(os.listdir(dir_point))
    # fns=[i[0:-4] for i in fns]
    fns=[fn for fn in fns if ((fn[0:-4] in test_ids))]
    fns=[i[0:-4] for i in fns]

print(cat[class_choice])
print(class_choice)
print(len(fns))

# check if the the paths exist
mesh_fns=[]
for ids in fns:
    print(os.path.join(v2_path,cat[class_choice],ids))
    if (os.path.exists(os.path.join(v2_path,cat[class_choice],ids))):
        if (os.path.exists(os.path.join(v2_path,cat[class_choice],ids,"models","pointcloud_final.npz"))):
            mesh_fns.append(ids)


print(len(mesh_fns))
print(len(fns))

def load_pretrain(model,pretrain):
    state_dict=torch.load(pretrain)
    new_state_dict=OrderedDict()
    for key,val in state_dict.items():
        name=key
        new_state_dict[name]=val
    model.load_state_dict(new_state_dict)
    return (model)


cfg=config.load_config("/vinai/sskar/convolutional_occupancy_networks/configs/pointcloud/shapenet_grid32.yaml",'configs/default.yaml')

decoder=cfg['model']['decoder']
encoder=cfg['model']['encoder']
dim=cfg['data']['dim']
c_dim=cfg['model']['c_dim']
decoder_kwargs=cfg['model']['decoder_kwargs']
encoder_kwargs=cfg['model']['encoder_kwargs']
padding=cfg['data']['padding']

encoder=encoder_dict[encoder](dim=dim,c_dim=c_dim,padding=padding,**encoder_kwargs)
decoder=models.decoder_dict[decoder](dim=dim, c_dim=c_dim, padding=padding,**decoder_kwargs)
model=models.ConvolutionalOccupancyNetwork(decoder,encoder,device=device)


#reconstruction weights
loss_func=ChamferLoss()
model_recon=ReconstructionNet().to(device)
model_recon=load_pretrain(model_recon,pretrain="/vinai/sskar/cvpr22/models/shapenetcorev2_250.pkl")
model_recon.eval()


weights=torch.load("./weights/shapenet_grid32.pt",map_location="cpu")
model.load_state_dict(weights["model"])

generator=config.get_generator(model,cfg,device=device)
# print(generator)

generate_mesh=cfg['generation']['generate_mesh']
generate_pointcloud=cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh=False
    print('Warning: generator does not support mesh generation.')
print("ok")
model.eval()

def eval_points(p,c=None,vol_bound=None):
        p_split = torch.split(p,100000)
        # print(p_split[0].shape)
        occ_hats=[]
        for pi in p_split:
            pi=pi.unsqueeze(0).to(device)
            with torch.no_grad():
                occ_hat=model.decoder(pi,c)
                p_r=dist.Bernoulli(logits=occ_hat)
                occ_hat=p_r.logits
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

def extract_mesh(mode,occ_hat,c,threshold,padding):
    n_x, n_y, n_z=occ_hat.shape
    box_size=1+padding
    threshold=np.log(threshold) - np.log(1.-threshold)
    occ_hat_padded=np.pad(occ_hat,1,'constant',constant_values=-1e6)
    vertices,triangles=libmcubes.marching_cubes(occ_hat_padded, threshold)  
    vertices -= 0.5
    vertices -= 1
    vertices/=np.array([n_x-1, n_y-1, n_z-1])
    vertices= box_size * (vertices-0.5)
    mesh = trimesh.Trimesh(vertices,triangles,vertex_normals=None,process=False)
    return (mesh)

def mise(threshold,padding,resolution0,upsampling_steps,mode):
    threshold1=np.log(threshold) - np.log(1. - threshold)
    box_size=1+padding
    mesh_extractor=MISE(resolution0,upsampling_steps,threshold1)
    points=mesh_extractor.query()
    while points.shape[0] != 0:
        pointsf=points / mesh_extractor.resolution
        # Normalize to bounding box 
        pointsf=box_size * (pointsf-0.5)
        pointsf=torch.FloatTensor(pointsf).to(device)
        # Evaluate model and update
        values=eval_points(pointsf,enc_op).cpu().numpy()
        values=values.astype(np.float64) 
        mesh_extractor.update(points, values)
        points=mesh_extractor.query()
    value_grid=mesh_extractor.to_dense()
    mesh=extract_mesh(mode,value_grid,enc_op,threshold,padding)
    return (mesh)



#load data
#mesh_fns:list of airplane ids
# data_out=data.copy()
# points=data[None]
# normals=data["normals"]
# indices=np.random.randint(points.shape[0],size=self.N)
# data_out[None]=points[indices,:]
# data_out["normals"]=normals[indices,:]
# return (data_out)


# """reconstruction pc and mesh for original pc"""
# for idx in mesh_fns:
#     #load pointcloud and sample 3000 points
#     point_path=os.path.join(v2_path,cat[class_choice],idx,"models","pointcloud_final.npz")
#     point_dict=np.load(point_path)  
#     pointcloud=point_dict["points"]
#     indices=np.random.randint(pointcloud.shape[0],size=2048)
#     pointcloud=pointcloud[indices,:]
#     pointcloud=pointcloud.astype(np.float32)


#     #saving pointclouds as ply files
#     # pcd=o3d.geometry.PointCloud()
#     # pcd.points=o3d.utility.Vector3dVector(pointcloud)
#     # o3d.io.write_point_cloud("./v0_test_pc/"+str(class_choice)+"/"+str(idx)+".ply",pcd)
 

#     # point cloud reconsturction
#     pts=torch.from_numpy(pointcloud).unsqueeze(0).to(device)
#     reconstructed_pl,_=model_recon(pts.view(1,pts.shape[1],3),sigma=0)
#     # reconstructed_pl+=torch.tensor(np.random.normal(0,0.05,size=reconstructed_pl.shape)).to(device)
#     loss_val=loss_func(pts.view(1,pts.shape[1],3),reconstructed_pl)
#     print(loss_val)
#     pcd=o3d.geometry.PointCloud()
#     pcd.points=o3d.utility.Vector3dVector(reconstructed_pl[0].detach().cpu().numpy())
#     o3d.io.write_point_cloud("./v0_plain_recon/"+str(class_choice)+"/"+str(idx)+".ply",pcd)
#     print("reconstruction saved")


# #     #mesh generation
# #     # with torch.no_grad():
# #     #     enc_op=model.encoder(torch.from_numpy(pointcloud).unsqueeze(0).to(device))
    
# #     # threshold=0.2
# #     # padding=0.1
# #     # resolution0=32
# #     # upsampling_steps=2
# #     # op_mesh=mise(threshold,padding,resolution0,upsampling_steps,mode="default")
# #     # mesh_out_file="./v0_test_meshes/"+str(class_choice)+"/"+str(idx)+".off"
# #     # op_mesh.export(mesh_out_file)
# #     # print("v0 mesh saved")


"""mesh for recon pc"""
recon_root_path="/vinai/sskar/mise/v0_noise_recon/"
recon_root_path1="/vinai/sskar/mise/v0_plain_recon/"


for idx in mesh_fns:
    #noise recon mesh
    # point_path=os.path.join(recon_root_path,str(class_choice),idx+".ply")
    # pcd=o3d.io.read_point_cloud(point_path)
    # pointcloud=np.asarray(pcd.points).astype(np.float32)


    # with torch.no_grad():
    #     enc_op=model.encoder(torch.from_numpy(pointcloud).unsqueeze(0).to(device))
    
    # threshold=0.2
    # padding=0.1
    # resolution0=32
    # upsampling_steps=2
    # op_mesh=mise(threshold,padding,resolution0,upsampling_steps,mode="default")
    # mesh_out_file="./v0_noise_recon_meshes/"+str(class_choice)+"/"+str(idx)+".off"
    # op_mesh.export(mesh_out_file)
    # print("v0 mesh saved")


    # plain recon mesh
    point_path=os.path.join(recon_root_path1,str(class_choice),idx+".ply")
    print(point_path)
    pcd=o3d.io.read_point_cloud(point_path)
    pointcloud=np.asarray(pcd.points).astype(np.float32)


    with torch.no_grad():
        enc_op=model.encoder(torch.from_numpy(pointcloud).unsqueeze(0).to(device))
    
    threshold=0.2
    padding=0.1
    resolution0=32
    upsampling_steps=2
    op_mesh=mise(threshold,padding,resolution0,upsampling_steps,mode="default")
    mesh_out_file="./v0_plain_recon_meshes/"+str(class_choice)+"/"+str(idx)+".off"
    op_mesh.export(mesh_out_file)
    print("v0 mesh saved")


