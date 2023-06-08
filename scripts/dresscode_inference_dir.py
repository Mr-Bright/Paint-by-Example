import argparse, os, sys, glob

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
from ldm.data.dresscode_dataset import DressCodeTestDataset
wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# def load_model_from_config(config, ckpt, verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     import pdb; pdb.set_trace()
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)

#     model.cuda()
#     model.eval()
#     return model

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    # pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    # sd = pl_sd["state_dict"]
    # import pdb; pdb.set_trace()
    model = instantiate_from_config(config.model)
    model.load_state_dict(load_state_dict(ckpt))
    # m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)

    model.cuda()
    model.eval()
    return model


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict



def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def main():
    outdir = "sample_results/dresscode"
    ddim_steps = 50
    plms = False
    ddim_eta = 0.0
    fixed_code = None
    skip_save = False
    n_iter = 2
    H = 1024
    W = 768
    n_imgs = 100
    C = 4
    f = 8
    n_samples = 1
    n_rows = 0
    scale = 1.0

    config = "configs/dresscode_orgclip.yaml"
    ckpt = "/home/kwang/DD_project/Paint-by-Example/models/dresscode_orgclip/2023-05-31T11-24-34_dresscode_orgclip/checkpoints/model_final.pth"
    # ckpt = '/home/kwang/DD_project/Paint-by-Example/models/dresscode_orgclip/2023-05-31T11-24-34_dresscode_orgclip/checkpoints/last.ckpt/global_step135640/mp_rank_00_model_states.pt'

    seed = 42
    precision = "full"
    
    seed_everything(seed)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "source")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    precision_scope = autocast if precision=="autocast" else nullcontext

    test_dataset = DressCodeTestDataset()
    interval = len(test_dataset) // 4
    sub = [x*interval for x in range(4)]
    sub.append(len(test_dataset))
    inx = 0
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for index in range(sub[inx], sub[inx+1]):
                    item = test_dataset[index]

                    filename = item['filename']
                    print(filename)
                    mask_tensor = item['inpaint_mask'].unsqueeze(0)
                    ref_tensor = item['ref_imgs'].unsqueeze(0)
                    inpaint_image = item['inpaint_image'].unsqueeze(0)
                    image_tensor = item['GT'].unsqueeze(0)

                    test_model_kwargs={}
                    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
                    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
                    ref_tensor=ref_tensor.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = model.learnable_vector
                    
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x):
                        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
                        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
                        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
                        return x

                    if not skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            
                            all_img=[]
                            all_img.append(un_norm(image_tensor[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=ref_tensor
                            ref_img=Resize([H, W])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample)
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            # img.save(os.path.join(grid_path, 'grid-'+filename[:-4]+'_'+str(seed)+'.png'))
                            
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(result_path, filename+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, filename+'_'+str(seed)+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, filename+'_'+str(seed)+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, filename+'_'+str(seed)+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, filename+'_'+str(seed)+"_ref.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    
if __name__ == "__main__":
    main()