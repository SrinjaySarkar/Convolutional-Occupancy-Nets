import torch
import torch.nn as nn
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

class local_decoder(torch.nn.Module):
  def __init__(self,point_dim=3,feat_dim=128,leaky=False,sample_mode="bilinear",padding=0.1,mid_dim=256,n_blocks=5):
    super().__init__()
    self.c_dim=feat_dim
    self.n_blocks=n_blocks
    self.sample_mode=sample_mode
    self.padding=padding

    if leaky == "False":
      self.ac_fn=torch.nn.fuctional.relu
    else:
      self.ac_fn= lambda x : torch.nn.functional.leaky_relu(x,0.2)
    
    self.c_layer1=torch.nn.ModuleList([torch.nn.Linear(feat_dim,mid_dim) for _ in range(n_blocks)])
    self.p_layer1=torch.nn.Linear(point_dim,mid_dim)

    self.res_blocks=torch.nn.ModuleList([res_block(mid_dim,None,None) for _ in range(n_blocks)])
    self.out_layer=torch.nn.Linear(mid_dim,1)
  
  def get_feats(self,p,c):
    norm_cords=norm_3d_cord(p.clone(),padding=self.padding)
    norm_cords=norm_cords[:,:,None,None].float()
    vol_grid=2.0*norm_cords-1.0
    sample_point_feats=torch.nn.functional.grid_sample(c,vol_grid,align_corners=True,mode=self.sample_mode).squeeze(-1).squeeze(-1)
    return (sample_point_feats)

  def forward(self,p,c,**kwargs):
    psi_px=0
    grid_type=list(c.keys())
    if "grid" in grid_type :
      psi_px+=self.get_feats(p,c["grid"])
    #psi_px=psi_px.transpose(1,2)
    psi_px=psi_px.permute(0,2,1)
    p=p.float()
    p_op=self.p_layer1(p)

    for i in range(self.n_blocks):
      p_op+=self.c_layer1[i](psi_px)
      p_op=self.res_blocks[i](p_op)
    
    final_op=self.ac_fn(p_op)
    final_op=self.out_layer(final_op)

    return (final_op.squeeze(-1))