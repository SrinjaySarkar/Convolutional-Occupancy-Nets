import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
import torch.distributions

from encoder import local_pool_pn
from ref_encoder import LocalPoolPointnet
from decoder import local_decoder
from ref_decoder import LocalDecoder


class conv_occ_net(torch.nn.Module):
	def __init__(self,encoder,decoder,device):
		super().__init__()
		self.encoder=encoder
		self.decoder=decoder
		self.device=device

class Trainer():
	def __init__(self,model,optimizer,device,input_type="pointcloud",vis_dir=None,threshold=0.5,eval_sample=False):
		self.model=model
		self.optimizer=optimizer
		self.device=device
		self.input_type=input_type
		self.vis_dir=vis_dir
		self.threshold=threshold
		self.eval_sample=eval_sample
		if vis_dir is not None and not os.path.exists(vis_dir):
			os.makedirs(vis_dir)

	def train_step(self,batch_data):
		self.model.train()
		self.optimizer.zero_grad()
		device=self.device
		p=batch_data.get("points").to(device)
		occ=batch_data.get("points.occ").to(device)
		inputs=batch_data.get("inputs").to(device)
		encoded_input=self.model.encoder(inputs)
		decoded_op=self.model.decoder(p,encoded_input)
		logits=torch.distributions.Bernoulli(logits=decoded_op).logits
		loss=F.binary_cross_entropy_with_logits(logits,occ,reduction="none")
		total_loss=loss.sum(-1).mean()
		total_loss.backward()
		self.optimizer.step()

		return total_loss.item()
	def val_step(self,batch_data):
		self.model.eval()

		device = self.device
		threshold = self.threshold
		eval_dict = {}

		points=batch_data.get("points").to(device)
		occ=batch_data.get("points.occ").to(device)
		inputs=batch_data.get("inputs").to(device)

		points_iou=batch_data.get("points_iou").to(device)
		occ_iou=batch_data.get("points_iou.occ").to(device)

		inputs=add_key(inputs, batch_data.get('inputs.ind'), 'points', 'index', device=device)
		points=add_key(points, batch_data.get('points.normalized'), 'p', 'p_n', device=device)
		points_iou=add_key(points_iou, batch_data.get('points_iou.normalized'),'p','p_n', device=device)

		with torch.no_grad():
			c=self.model.encoder(inputs)
			p_r=self.model.decoder(points_iou,c)
			p_r=torch.distributions.Bernoulli(logits=p_r)

		occ_iou_np=(occ_iou >= 0.5).cpu().numpy()
		occ_iou_hat_np=(p_r.probs >= threshold).cpu().numpy()

		iou=compute_iou(occ_iou_np,occ_iou_hat_np).mean()
		eval_dict["iou"]=iou

		return (eval_dict)



