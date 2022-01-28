import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from unet_encoder import UNet3D
from torch_scatter import scatter_mean,scatter_max
import torch.nn.functional as F

def norm_3d_cord(p,padding=0.1):
  norm_cord= p / (1 + padding + 10e-4) # (-0.5, 0.5)
  #norm_cord=p/(1+padding+10e-4)
  #print("after step 1:",norm_cord)
  norm_cord=norm_cord+ 0.5
  #print("after step 2:",norm_cord)
  if norm_cord.max() >=1 :
    norm_cord[norm_cord >= 1]=1-10e-4
  if norm_cord.min() < 0 :
    norm_cord[norm_cord < 0]=0.0
  #print("after step 3:",norm_cord)
  return (norm_cord)

def cord2index(p,reso,coord_type):
  #assigns an index(based on x,y,z value) to each point in the point cloud of each batch
  p=p*reso
  p=p.long()
  if coord_type=="2d":
    idx=p[:,:,0]+reso*p[:,:,1]
  if coord_type=="3d":
    idx=p[:,:,0]+reso*(p[:,:,1]+reso*p[:,:,2])
  idx=idx.unsqueeze_(1)
  return (idx)

class res_block(torch.nn.Module):
  def __init__(self,in_dim,mid_dim=None,out_dim=None):
    super().__init__()
    if out_dim is None:
      out_dim=in_dim
    if mid_dim is None:
      mid_dim=min(in_dim,out_dim)
    self.in_dim=in_dim
    self.mid_dim=mid_dim
    self.out_dim=out_dim

    self.fc1=torch.nn.Linear(in_dim,mid_dim)
    self.fc2=torch.nn.Linear(mid_dim,out_dim)

    if in_dim==out_dim:
      self.fc3=None
    else:
      self.fc3=torch.nn.Linear(in_dim,out_dim,bias=False)
    torch.nn.init.zeros_(self.fc2.weight)
  
  def forward(self,x):
    x1=F.relu(x)
    x1=self.fc1(x1)
    x1=F.relu(x1)
    x1=self.fc2(x1)

    if self.fc3 is not None:
      xs=self.fc3(x)
    else:
      xs=x
    op=xs+x1
    return (op)

class local_pool_pn(torch.nn.Module):
  def __init__(self,c_dim,mid_dim,scatter_type,unet3d,unet3d_kwargs,grid_resolution,padding,n_blocks):
    super().__init__()
    self.c_dim=c_dim
    self.mid_dim=mid_dim
    self.reso_grid=grid_resolution
    self.padding=padding
    
    self.layer1=torch.nn.Linear(3,2*mid_dim)
    self.res_blocks=torch.nn.ModuleList([res_block(in_dim=2*mid_dim,mid_dim=None,out_dim=mid_dim) for _ in range(n_blocks)])
    self.layer2=torch.nn.Linear(mid_dim,c_dim)
    
    if unet3d == True:
      self.unet3d=UNet3D(**unet3d_kwargs)
    else:
      self.unet3d=None
    if scatter_type == "max":
      self.scatter=scatter_max
    elif scatter_type == "mean":
      self.scatter= scatter_mean
    else:
      raise ValueError("wrong scatter")

    
    
  
  def grid_feats(self,p,c):
    p1=p
    norm_cords=norm_3d_cord(p1,padding=self.padding)
    idx=cord2index(norm_cords,self.reso_grid,coord_type="3d")
    ## at this point we have normalized cordinates and the indices for each point in the PC
    feat_grid=c.new_zeros(p.shape[0],self.c_dim,self.reso_grid**3)
    # feature grid shape : (bs,c_dim,32*32*32)
    c=c.permute(0,2,1)
    #scatter the features of the points to find the features of the points in the grid by averaging the existing features
    feat_grid=scatter_mean(c,idx,out=feat_grid)#bs*c_dim*reso**3
    feat_grid=feat_grid.reshape(p.shape[0],self.c_dim,self.reso_grid,self.reso_grid,self.reso_grid)
    # the total number of points are res**3 each with a feature dimension of c_dim 

    if self.unet3d is not None:
      grid_feat=self.unet3d(feat_grid)
    return (grid_feat)
  
  def pool_local_feats(self,xy,index,c):
    #this function is used for pooling features in the localilty of the points instead of the global pooling in pointntet
    bs,feat_dim=c.shape[0],c.shape[2]
    keys=xy.keys()
    op=0
    for key in keys:
      if key=="grid":
        feat=scatter_mean(c.permute(0,2,1),index[key],dim_size=self.reso_grid**3)
      else:
        feat=scatter_mean(c.permute(0,2,1),index[key],dim_size=self.reso_grid**3)
      feat=feat.gather(dim=2,index=index[key].expand(-1,feat_dim,-1))
      op+=feat
    op=op.permute(0,2,1)
    return (op)

  def forward(self,p):
    bs,n,d=p.shape[0],p.shape[1],p.shape[2]
    coord={}
    idx={}
    feat={}
    coord["grid"]=norm_3d_cord(p.clone(),padding=self.padding)
    idx["grid"]=cord2index(coord["grid"],self.reso_grid,coord_type="3d")

    p1=self.layer1(p)
    p1=self.res_blocks[0](p1)
    for block in self.res_blocks[1:]:
      pooled_feats=self.pool_local_feats(coord,idx,p1)
      p1=torch.cat([p1,pooled_feats],dim=2)
      p1=block(p1)
    
    inter_feats=self.layer2(p1)
    feat["grid"]=self.grid_feats(p,inter_feats)

    return (feat)   

