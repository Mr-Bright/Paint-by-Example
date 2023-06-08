python -u main.py --logdir models/dresscode_classlabel --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/dresscode_orgclip_classlabel.yaml --scale_lr False

python -u main.py --logdir models/dresscode_conv --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/dresscode.yaml --scale_lr False

python -u main.py --logdir models/dresscode_orgclip --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/dresscode_orgclip.yaml --scale_lr False -r /home/kwang/DD_project/Paint-by-Example/models/dresscode_orgclip/2023-05-31T11-24-34_dresscode_orgclip

python -u main.py --logdir models/dresscode_fullclip --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/dresscode_fullclip.yaml --scale_lr False


# 用autoencoder作为condition输入
python -u main.py --logdir models/dresscode_autoencoder --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/dresscode_autoencoder_cloth.yaml --scale_lr False


# 将warped_clothes也加入输入中，condition输入用zeros代替
python -u main.py --logdir models/dresscode_13channels --pretrained_model pretrained_models/sd-v1-4-modified-13channel.ckpt --base configs/dresscode_13channels.yaml --scale_lr False
