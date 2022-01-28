import os 
import torch
import h5py
import json
import numpy as np
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from encoder import local_pool_pn
from ref_encoder import LocalPoolPointnet
from decoder import local_decoder
from ref_decoder import LocalDecoder
from seg_trainer import Trainer
from seg_trainer import conv_seg_net

def collate_remove_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return utils.data.dataloader.default_collate(batch)

class segment_Dataset(utils.data.Dataset):
	def __init__(self,root_path,split,n_points=2500,class_choice=None,data_augmentation=False):
		self.npoints=n_points
		self.root_path=root_path
		self.catfile=os.path.join(root_path,"synsetoffset2category.txt")
		self.category_dict={}
		self.data_augmentation=data_augmentation
		with open(self.catfile,"r") as f:
			for line in f:
				ls=line.strip().split()
				self.category_dict[ls[0]]=ls[1]	
		if not class_choice is None:
			self.category_dict={k: v for k, v in self.category_dict.items() if k in class_choice}
		self.id2cat={v:k for k,v in self.category_dict.items()}
		self.meta={}
		splitfile=os.path.join(root_path, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
		filelist=json.load(open(splitfile,"r"))
		for item in self.category_dict:
			self.meta[item]=[]
		for file in filelist:
			_,category,uuid=file.split("/")
			if category in self.category_dict.values():
				#print(category)
				self.meta[self.id2cat[category]].append((os.path.join(root_path, category, 'points', uuid+'.pts'),os.path.join(root_path, category, 'points_label', uuid+'.seg')))
		self.datapath=[]
		for item in self.category_dict:
			for fn in self.meta[item]:
				self.datapath.append((item,fn[0],fn[1]))
		self.classes=dict(zip(sorted(self.category_dict), range(len(self.category_dict))))



	def __getitem__(self,idx):
		fn=self.datapath[idx]
		cls=self.classes[self.datapath[idx][0]]
		#print(cls)

		point_set=np.loadtxt(fn[1]).astype(np.float32)
		seg=np.loadtxt(fn[2]).astype(np.int64)

		choice=np.random.choice(len(seg),self.npoints,replace=True)
		point_set=point_set[choice,:]
		point_set=point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
		dist=np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
		point_set=point_set / dist
		if self.data_augmentation:
			theta=np.random.uniform(0,np.pi*2)
			rotation_matrix=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
			point_set[:,[0,2]]=point_set[:,[0,2]].dot(rotation_matrix) # random rotation
			point_set+=np.random.normal(0, 0.02, size=point_set.shape) # random jitter
		seg=seg[choice]
		
		point_set=torch.from_numpy(point_set)
		seg=torch.from_numpy(seg)

		return (point_set,seg)
	
	def __len__(self):
		return (len(self.datapath))
#WHENEVER A CATEGORY CHANGES MAKE CHANGE IN DECODER.PY , TRAINER.PY
#self,root_path,n_points=2500,class_choice=None,split="train",data_augmentation=False
datapath="./partanno"
sample_seg_dataset=segment_Dataset(root_path=datapath,class_choice=["Chair"],split="test")
d=segment_Dataset(root_path=datapath,class_choice=["Lamp"],split='test',data_augmentation=False)
print(d[0][0].shape)
print(d[0][1].shape)


#dataloader
# train_loader=utils.data.DataLoader(sample_seg_dataset,batch_size=2,num_workers=4,shuffle=True)

# # ref_encoder=LocalPoolPointnet(c_dim=32, dim=3, hidden_dim=32, scatter_type='max', 
# # unet=False, unet_kwargs=None, unet3d=True, unet3d_kwargs={"num_levels":3,"f_maps":32,"in_channels":32,"out_channels":32}, plane_resolution=None,grid_resolution=32, plane_type='grid', padding=0.1, n_blocks=5)
# my_encoder=local_pool_pn(c_dim=32,mid_dim=32,scatter_type="max",unet3d=True,unet3d_kwargs={"num_levels":3,"f_maps":32,"in_channels":32,"out_channels":32},
#                          grid_resolution=32,padding=0.1,n_blocks=5)

# # ref_decoder=LocalDecoder(dim=3, c_dim=32,hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1)
# my_decoder=local_decoder(point_dim=3,feat_dim=32,leaky=False,sample_mode="bilinear",padding=0.1,mid_dim=32,n_blocks=5)


# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# my_model=conv_seg_net(encoder=my_encoder,decoder=my_decoder,device=device)
# opt=optim.Adam(my_model.parameters(), lr=1e-4) 

# trainer=Trainer(model=my_model,optimizer=opt,device=device,vis_dir=None,threshold=0.5,eval_sample=False)

# train_loader=utils.data.DataLoader(sample_seg_dataset,batch_size=2, num_workers=4, shuffle=False,collate_fn=collate_remove_none)
# for batch_idx,batch in enumerate(train_loader):
# 	if batch_idx==0:
# 		l=trainer.train_step(batch)
# 		print(l)
# 		lv=trainer.val_step(batch)
# 		print(lv)
# 	else:
# 		break
























