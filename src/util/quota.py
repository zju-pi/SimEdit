import torch
from torch.nn.functional import mse_loss
# from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from lpips import LPIPS

loss_fn_alex = LPIPS(net='vgg') # best forward scores

def pic2numpy(img: torch.Tensor):
    return img.permute(1,2,0).cpu().numpy()

def calc_mse(img1: torch.Tensor,img2: torch.Tensor):
    return mse_loss(img1, img2).item()

def calc_lpips(img1: torch.Tensor,img2: torch.Tensor):
    return loss_fn_alex(img1,img2).item()

def calc_ssim(img1: torch.Tensor,img2: torch.Tensor):
    return SSIM(pic2numpy(img1), pic2numpy(img2), multichannel=True, channel_axis=2,data_range=1)

def calc_psnr(img1: torch.Tensor,img2: torch.Tensor):
    return PSNR(pic2numpy(img1),pic2numpy(img2))