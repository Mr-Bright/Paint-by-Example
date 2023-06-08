import torch
ckpt = '/home/kwang/DD_project/Paint-by-Example/models/dresscode_orgclip/2023-05-31T11-24-34_dresscode_orgclip/checkpoints/model_final.pth'


# from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# # lightning deepspeed has saved a directory instead of a file
# output_path = "lightning_model.ckpt"
# convert_zero_checkpoint_to_fp32_state_dict(ckpt, output_path)

pl_sd = torch.load(ckpt, map_location='cpu')

print(pl_sd.keys())
print(pl_sd['pytorch-lightning_version'])