import os
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
import torch.distributions

from encoder import local_pool_pn
from ref_encoder import LocalPoolPointnet
from decoder import local_decoder
from ref_decoder import LocalDecoder

class conv_seg_net(torch.nn.Module):
	def __init__(self,encoder,decoder,device):
		super().__init__()
		self.encoder=encoder
		self.decoder=decoder
		self.device=device

class Trainer():
	def __init__(self,model,optimizer,device,vis_dir=None,threshold=0.5,eval_sample=False):
		self.model=model
		self.optimizer=optimizer
		self.device=device
		self.vis_dir=vis_dir
		self.threshold=threshold
		self.eval_sample=eval_sample
		if vis_dir is not None and not os.path.exists(vis_dir):
			os.makedirs(vis_dir)
	
	def train_step(self,batch_data):
		self.model.train()
		self.optimizer.zero_grad()
		my_encoder=self.model.encoder
		my_decoder=self.model.decoder
		device=self.device
		
		points=batch_data[0]
		seg_op=batch_data[1]

		#encoder,decoder
		encoded_outut=my_encoder(points)
		decoded_op=my_decoder(points,encoded_outut)
		
		z=F.log_softmax(decoded_op,-1)
		re_dec1=z.view(-1,4)
		y1=seg_op.view(-1, 1)[:, 0] - 1
		
		loss=F.nll_loss(re_dec1,y1)
		loss.backward()

		self.optimizer.step()

		return (loss.item())

	def val_step(self,batch_data):
		shape_ious=[]
		my_encoder=self.model.encoder
		my_decoder=self.model.decoder
		self.model.eval()
		# multi class iou

		points=batch_data[0]
		seg_op=batch_data[1]

		#encoder,decoder
		encoded_outut=my_encoder(points)
		decoded_op=my_decoder(points,encoded_outut)

		##
		z=F.softmax(decoded_op,-1)
		pred_choice=z.data.max(2)[1]
		pred_np=pred_choice.cpu().data.numpy()
		target_np=seg_op.cpu().data.numpy() - 1

		for shape_idx in range(target_np.shape[0]):
			parts=range(4)
			part_ious=[]
			for part in parts:
				intersection=np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
				union=np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
				if union==0:
					iou=1
				else:
					iou=intersection/float(union)
				part_ious.append(iou)
			shape_ious.append(np.mean(part_ious))
		return (np.mean(shape_ious))

