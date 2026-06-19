from PIL import Image
import numpy as np
from einops import rearrange
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch.nn as nn
import torch
import cv2
from typing import Tuple, Union, List
from transformers import T5TokenizerFast
from sklearn.cluster import KMeans
from torchvision import transforms as T
import matplotlib.colorizer as mcolorizer

COLOR_MAP = ['#A50026','#384C9F','#DC3B2C','#5E93C3','#9DCEE2','#F7874E','#DEF2F7','#FDCC7D','#FEF9B6',]

def hex_to_rgb(color):
    return int(color[1:3],16), int(color[3:5],16), int(color[5:7], 16)

def matplotlib_colored_img(arr, vmin=None, vmax=None, cmap=None):
    sm = mcolorizer.Colorizer(cmap=cmap)
    sm.set_clim(vmin, vmax)
    rgba = sm.to_rgba(arr, bytes=True)
    pil_shape = (rgba.shape[1], rgba.shape[0])
    rgba = np.require(rgba, requirements='C')
    image = Image.frombuffer("RGBA", pil_shape, rgba, "raw", "RGBA", 0, 1)
    return image

def pooling(attn,height,pool=2,stride=2):
    pool = nn.MaxPool2d(pool, stride=stride)

    h = attn.shape[1]
    attn = rearrange(attn,'b h (H W) d -> b (h d) H W',H = height)

    attn = pool(attn)
    height = attn.shape[2]
    return rearrange(attn,'b (h d) H W -> b h (H W) d',h = h), height

def pca_visualize(data,height,channel=3,blur=False,resize=True):
    """
    pca visualize function, refer from plug-and-play
    """
    width = data.shape[1] // height
    
    res = []
    for img in tqdm(data):
        print(img.shape)
        pca = PCA(n_components=channel,random_state=42)
        pca.fit(img)
        pca_img = pca.transform(img) # [n,channel]
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) * 255 / (pca_img_max - pca_img_min)
        pca_img = pca_img.reshape(height,width,channel).astype(np.uint8)
        if channel == 1: pca_img = pca_img.squeeze()
        pca_img = Image.fromarray(pca_img)
        if resize:
            if blur:
                pca_img = pca_img.resize((width*16,height*16))
            else:
                pca_img = T.Resize((height*16,width*16),interpolation=T.InterpolationMode.NEAREST)(pca_img)
        res.append(pca_img)
    return res

def kmeans_visualize(data,height,channel,blur=False,resize=True):
    width = data.shape[1] // height

    res = []
    for img in tqdm(data):
        kmeans = KMeans(n_clusters=channel,random_state=42)
        kmeans.fit(img)
        R = []
        G = []
        B = []
        for i in kmeans.labels_:
            r, g, b = hex_to_rgb(COLOR_MAP[i])
            R.append(r)
            G.append(g)
            B.append(b)
        kmeans_img = np.array([R,G,B]).T
        kmeans_img = kmeans_img.reshape(height,width,3).astype(np.uint8)
        kmeans_img = Image.fromarray(kmeans_img)
        if resize:
            if blur:
                kmeans_img = kmeans_img.resize((width*16,height*16))
            else:
                kmeans_img = T.Resize((height*16,width*16),interpolation=T.InterpolationMode.NEAREST)(kmeans_img)
        res.append(kmeans_img)
    return res

def avg_visualize(data,height,blur=False,resize=True,norm=True):
    width = data.shape[1] // height
    res = []
    for img in tqdm(data):
        avg_img = np.array(img.mean(1).to(torch.float32))
        if norm:
            avg_img_min = avg_img.min()
            avg_img_max = avg_img.max()
            avg_img = (avg_img - avg_img_min) * 255 / (avg_img_max - avg_img_min)
        avg_img = avg_img.reshape(height,width).astype(np.uint8)
        avg_img = Image.fromarray(avg_img)
        if resize:
            if blur:
                avg_img = avg_img.resize((width*16,height*16))
            else:
                avg_img = T.Resize((height*16,width*16),interpolation=T.InterpolationMode.NEAREST)(avg_img)
        res.append(avg_img)
    return res

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    add text under a image function, refer from prompt-to-prompt
    """
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

TOKEN_LEN = 512

def view_self_attn(attn: torch.Tensor,height,channel=3,type='pca',pool=2,stride=2,blur=False):
    # cross_attn = attn[:,:,TOKEN_LEN:,:TOKEN_LEN].cpu().float()
    self_attn = attn[:,:,TOKEN_LEN:,TOKEN_LEN:].cpu().float()
    
    height = height // 16
    if type == 'pca':
        print('pooling to pca...')
        attn, height = pooling(self_attn,height,pool,stride)

    vis = pca_visualize if type == 'pca' else kmeans_visualize

    return vis(rearrange(attn,'b h n d -> b n (h d)'),height,channel,blur)

def view_cross_attn(attn: torch.Tensor,prompts: Union[List[str],str],tokenizer: T5TokenizerFast,height,channel=1,blur=False,pool=2,stride=2,head='pca',add_text=True,cross_idx=0):
    if isinstance(prompts,str): prompts = [prompts]
    
    cross_attn = attn[:,:,TOKEN_LEN:,:TOKEN_LEN].cpu().float() if cross_idx == 0 else attn[:,:,:TOKEN_LEN,TOKEN_LEN:].cpu().transpose(2,3).float()
    tokens = []
    for prompt in prompts:
        tokens.append(tokenizer.encode(prompt))
    decoder = tokenizer.decode

    height = height // 16

    attn, height = pooling(cross_attn,height,pool,stride)
    attn = rearrange(attn,'b h n d -> b d n h')

    def get_imgs(attn,tokens):
        if head == 'pca':
            imgs = pca_visualize(attn,height,channel,blur)
        else:
            imgs = avg_visualize(attn,height,blur)
        if add_text:
            for i, img in enumerate(imgs):
                img = text_under_image(np.array(img)[:,:,None] if channel==1 or head=='avg' else np.array(img),decoder(int(tokens[i])))
                imgs[i] = img
        imgs = np.hstack(imgs).squeeze()
        return Image.fromarray(imgs)

    res = []
    for i,_ in enumerate(prompts):
        res.append(get_imgs(attn[i,:len(tokens[i])],tokens[i]))
    
    if len(res) == 1: return res[0]
    return res