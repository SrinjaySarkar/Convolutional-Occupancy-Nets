import os
import torch
import h5py
import time
import json
from collections import defaultdict
import numpy as np
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
##seg
from seg_encoder import local_pool_pn
from seg_ref_encoder import LocalPoolPointnet
from seg_decoder import local_decoder
from seg_ref_decoder import LocalDecoder
from seg_trainer import Trainer
from seg_trainer import conv_seg_net


from seg_data import segment_Dataset
datapath="./shapenetcore_partanno_segmentation_benchmark_v0"
segmentation_dataset_train=segment_Dataset(root_path=datapath,class_choice=["Pistol"],split="train",data_augmentation=True)
segmentation_dataset_val=segment_Dataset(root_path=datapath,class_choice=["Pistol"],split="test",data_augmentation=True)

train_loader=utils.data.DataLoader(segmentation_dataset_train,batch_size=32,num_workers=4,shuffle=True)
val_loader=utils.data.DataLoader(segmentation_dataset_val,batch_size=32,num_workers=4,shuffle=True)

seg_ref_encoder=LocalPoolPointnet(c_dim=32, dim=3, hidden_dim=32, scatter_type='max',unet=False,
unet_kwargs=None, unet3d=True, unet3d_kwargs={"num_levels":3,"f_maps":32,"in_channels":32,
"out_channels":32}, plane_resolution=None,grid_resolution=32, plane_type='grid', padding=0.1,
n_blocks=4)
seg_ref_decoder=LocalDecoder(dim=3, c_dim=32,hidden_size=32, n_blocks=4, leaky=False,
                         sample_mode='bilinear', padding=0.1)

trainer=Trainer(model=my_model,optimizer=opt,device=device,vis_dir=None,threshold=0.5,eval_sample=False)

n_epochs=500
for epoch in range(n_epochs):
  my_model.train()
  # if epoch % 10 == 0 and epoch > 0 :
  #   scheduler.step()
  for batch_idx,batch in enumerate(train_loader):

    points=batch[0].to(device)
    seg_op=batch[1].to(device)
    #encoder,decoder
    encoded_output=my_model.encoder(points)
    decoded_op=my_model.decoder(points,encoded_output)
    z=F.log_softmax(decoded_op,-1)
    re_dec1=z.view(-1,3)
    y1=seg_op.view(-1, 1)[:, 0] - 1
    loss=F.nll_loss(re_dec1,y1)
    loss.backward()
    opt.step()
    if batch_idx % 50 == 0:
      print("Training Loss for epoch %d == %.3f" % (epoch,loss))
      pred_choice=re_dec1.data.max(1)[1]
      correct=pred_choice.eq(y1.data).cpu().sum()
      print('[%d: %d] train loss: %f accuracy: %f' % (epoch,batch_idx, loss.item(), correct.item()/float(32 * 2500)))



  if epoch% 10 == 0:
    torch.save(my_model.state_dict(), 'seg_model_%s_%d.pth' % ("Rocket",epoch))
    ##val
    shape_ious=[]
    my_model.eval()
    for batch_idx,batch in enumerate(val_loader):

      points=batch[0].to(device)
      seg_op=batch[1].to(device)
      #encoder,decoder
      encoded_output=my_model.encoder(points)
      decoded_op=my_model.decoder(points,encoded_output)
      z=F.log_softmax(decoded_op,-1)
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
    print("Mean IOU for shape on validation set:",np.mean(shape_ious))
