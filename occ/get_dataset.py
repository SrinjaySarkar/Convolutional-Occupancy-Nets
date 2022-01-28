import os
import logging
import torch
from torchvision import transforms 
import numpy as np
from src import data 
from src.common import add_key
import torch.utils as utils
import yaml
from tqdm import tqdm


import torch.optim as optim
from trainer import Trainer
from trainer import conv_occ_net
from encoder import local_pool_pn
from ref_encoder import LocalPoolPointnet
from decoder import local_decoder
from ref_decoder import LocalDecoder

def load_config(path, default_path=None):
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)
    inherit_from = cfg_special.get('inherit_from')
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()
    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def collate_remove_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return utils.data.dataloader.default_collate(batch)


def get_data_fields(mode, cfg):
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(cfg['data']['points_file'], points_transform,unpackbits=cfg['data']['points_unpackbits'],multi_files=cfg['data']['multi_files'])
        else:
            fields['points'] = data.PatchPointsField(cfg['data']['points_file'], transform=points_transform,unpackbits=cfg['data']['points_unpackbits'],multi_files=cfg['data']['multi_files'])
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(points_iou_file,unpackbits=cfg['data']['points_unpackbits'],multi_files=cfg['data']['multi_files'])
            else:
                fields['points_iou'] = data.PointsField(points_iou_file,unpackbits=cfg['data']['points_unpackbits'],multi_files=cfg['data']['multi_files']) 
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)
    return (fields)

#get data fields test
##print(fields["points_iou.occ"])

def get_inputs_field(mode, cfg):
    input_type = cfg['data']['input_type']
    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([data.SubsamplePointcloud(cfg['data']['pointcloud_n']),data.PointcloudNoise(cfg['data']['pointcloud_noise'])])
        inputs_field = data.PointCloudField(cfg['data']['pointcloud_file'], transform,multi_files= cfg['data']['multi_files'])
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([data.SubsamplePointcloud(cfg['data']['pointcloud_n']),data.PointcloudNoise(cfg['data']['pointcloud_noise'])])
        inputs_field = data.PartialPointCloudField(cfg['data']['pointcloud_file'], transform,multi_files= cfg['data']['multi_files'])
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([data.SubsamplePointcloud(cfg['data']['pointcloud_n']),data.PointcloudNoise(cfg['data']['pointcloud_noise'])])
        inputs_field = data.PatchointCloudField(cfg['data']['pointcloud_file'], transform,multi_files= cfg['data']['multi_files'])
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(cfg['data']['voxels_file'])
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError("Wrong input type (%s)" % input_type)  
    return (inputs_field)
#get input field test


# inputs_field=get_inputs_field(mode="val",cfg=cfg1)
# #print(inputs_field)

# if inputs_field is not None:
#     fields['inputs'] = inputs_field
# #print(fields)
###################################GET_DATA########################################
class Shapes3dDataset(utils.data.Dataset):
    def __init__(self, dataset_folder, fields, split=None,categories=None, no_except=True, transform=None, cfg=None):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))]
        metadata_file=os.path.join(dataset_folder,'metadata.yaml')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {c: {'id': c, 'name': 'n/a'} for c in categories} 
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)
            if split is None:
                self.models += [{'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
    def __len__(self):
        length=len(self.models)
        return (length)

    def __getitem__(self,idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']
        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}
        if self.cfg['data']['input_type'] == 'pointcloud_crop':
            info = self.get_vol_info(model_path)
            data['pointcloud_crop'] = True
        else:
            info = c_idx
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, info)
            except Exception:
                if self.no_except:
                    logger.warn('Error occured when loading field %s of model %s'% (field_name, model))
                    return None
                else:
                    raise
            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self,idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s' %(field_name, model_path))
                return False
        return True

########################################################################dataloader#################################################################

mode="val"
cfg1=load_config("./shapenet_grid32.yaml","./default.yaml")

method = cfg1['method']
dataset_type = cfg1['data']['dataset']
dataset_folder = cfg1['data']['path']
categories = cfg1['data']['classes']

splits = {'train': cfg1['data']['train_split'],'val': cfg1['data']['val_split'],'test': cfg1['data']['test_split'],}
split = splits[mode]
return_idx=True
if dataset_type == 'Shapes3D':
    fields=get_data_fields(mode,cfg1)
    inputs_field = get_inputs_field(mode, cfg1)
    if inputs_field is not None:
        fields['inputs'] = inputs_field
    if return_idx:
        fields['idx'] = data.IndexField()
val_dataset=Shapes3dDataset(dataset_folder, fields,split=split,categories=categories,cfg = cfg1)
####################################################################################################################################################
train_loader=utils.data.DataLoader(val_dataset, batch_size=2, num_workers=cfg1['training']['n_workers_val'], shuffle=False,collate_fn=collate_remove_none)
########################encoder#########################################################################################################################
ref_encoder=LocalPoolPointnet(c_dim=32, dim=3, hidden_dim=32, scatter_type='max', 
unet=False, unet_kwargs=None, unet3d=True, unet3d_kwargs={"num_levels":3,"f_maps":32,"in_channels":32,"out_channels":32}, plane_resolution=None,grid_resolution=32, plane_type='grid', padding=0.1, n_blocks=5)
my_encoder=local_pool_pn(c_dim=32,mid_dim=32,scatter_type="max",unet3d=True,unet3d_kwargs={"num_levels":3,"f_maps":32,"in_channels":32,"out_channels":32},
                         grid_resolution=32,padding=0.1,n_blocks=5)
#####################################################################decoder###################################################################################
ref_decoder=LocalDecoder(dim=3, c_dim=32,hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1)
my_decoder=local_decoder(point_dim=3,feat_dim=32,leaky=False,sample_mode="bilinear",padding=0.1,mid_dim=32,n_blocks=5)
###############################################################################################################################################################

# for batch_idx,batch in enumerate(tqdm(train_loader)):
    
#     if batch_idx == 0:
    	
#     	print(batch.keys())
#     	points=batch.get("points")
#     	occp=batch.get("points.occ")
#     	inputs=batch.get("inputs")
#     	voxels_occ=batch.get("voxels")
    	
#     	## ENCODER
#     	#ref_op=ref_encoder(inputs)
#     	my_op=my_encoder(inputs)
#     	print("encoded shape")
#     	#print(ref_op["grid"].shape)
#     	print(my_op["grid"].shape)

#     	## DECODER
#     	#ref_opd=ref_decoder(points,ref_op)
#     	my_opd=my_decoder(points,my_op)
#     	p_r = torch.distributions.Bernoulli(logits=my_opd)
#     	print(p_r.probs)
#     	print("decoded shape")
#     	#print(ref_opd.shape)
#     	print(my_opd.shape)
#     else: 
#     	break
    # print(points.shape)
#################################################################################TRAINING######################################################################
device=torch.device("cpu")
my_model=conv_occ_net(encoder=my_encoder,decoder=my_decoder,device=device)
opt=optim.Adam(my_model.parameters(), lr=1e-4) 
sample_trainer=Trainer(model=my_model,optimizer=opt,device=device,input_type="pointcloud",vis_dir=None,threshold=0.5,eval_sample=False)


for batch_idx,batch in enumerate(tqdm(train_loader)):
	if batch_idx==0:
		loss=sample_trainer.val_step(batch)
		print(loss["iou"])
	else:
		break
