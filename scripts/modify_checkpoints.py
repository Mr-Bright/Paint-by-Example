import torch
pretrained_model_path='pretrained_models/sd-v1-4.ckpt'
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')
# 改这儿可以改通道数
zero_data=torch.zeros(320,9,3,3)
new_weight=torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'],zero_data),dim=1)
ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight']=new_weight
torch.save(ckpt_file,"pretrained_models/sd-v1-4-modified-13channel.ckpt")