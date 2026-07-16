import os
import torch
import torchvision.transforms as T
from IPython.display import display
from ..model.attention import AttentionBase
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel
from src.model.diffuser import ParallelFluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxTransformer2DModel
 
def read_image(image_path,height=None,vae_scale_factor=16):
    img = Image.open(image_path).convert("RGB")
    owidth, oheight = img.size
    if height == None:
        width = owidth
        height = oheight
    else:
        width = int(height * owidth / oheight)
    width = width if width % vae_scale_factor == 0 else width - width % vae_scale_factor
    height = height if height % vae_scale_factor == 0 else height - height % vae_scale_factor
    img = T.Resize((height,width))(img)
    return T.ToTensor()(img)

def show_image(tensor: torch.Tensor):
    img = T.ToPILImage()(tensor.float())
    display(img)

def save_image(tensor: torch.Tensor,path: str):
    img = T.ToPILImage()(tensor.float())
    img.save(path)

def reset_attn(pipe):
    double_location = f"transformer_blocks.[{','.join([str(i) for i in range(19)])}].attn"
    single_location = f"single_transformer_blocks.[{','.join([str(i) for i in range(38)])}].attn"
    pipe.register_attn(double_location,AttentionBase())
    pipe.register_attn(single_location,AttentionBase())

def prompt_embd_aligned_replacement(target,src,tokenizer_2,reverse=False,padding_max_length = False,max_length = 512):
    target_id = tokenizer_2.encode(target,padding='max_length',max_length=512) if padding_max_length else tokenizer_2.encode(target)
    src_id = tokenizer_2.encode(src,padding='max_length',max_length=512) if padding_max_length else tokenizer_2.encode(src)
    
    idxs = {}
    cnt = {}
    length = len(target_id)
    for i,id in enumerate(reversed(target_id) if reverse else target_id):
        if id not in idxs:
            idxs[id] = []
            cnt[id] = 0
        idxs[id].append(length - i - 1 if reverse else i)

    src_replacement = []
    target_replacement = []
    for i,id in enumerate(src_id):
        if id not in idxs or cnt[id] >= len(idxs[id]): continue
        src_replacement.append(i)
        target_replacement.append(idxs[id][cnt[id]])
        cnt[id] += 1

    print(f'replace words: {tokenizer_2.decode(np.array(target_id)[target_replacement])}')
    # New: output each pair of index and its word
    for t_idx, s_idx in zip(target_replacement, src_replacement):
        t_word = tokenizer_2.decode([target_id[t_idx]])
        s_word = tokenizer_2.decode([src_id[s_idx]])
        print(f'target[{t_idx}]:"{t_word}" <-> source[{s_idx}]:"{s_word}"')
    
    return target_replacement, src_replacement

def prompt_embd_aligned_replacement_advanced(target, src, tokenizer_2, padding_max_length=False, max_length=512):
    """
    Use Longest Common Subsequence (LCS) algorithm to align tokens in source and target prompts, return corresponding token indices.
    """
    target_id = tokenizer_2.encode(target, padding='max_length', max_length=max_length) if padding_max_length else tokenizer_2.encode(target)
    src_id = tokenizer_2.encode(src, padding='max_length', max_length=max_length) if padding_max_length else tokenizer_2.encode(src)

    n, m = len(target_id), len(src_id)
    # Build LCS DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if target_id[i] == src_id[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    # Backtrack to find all matching token indices
    i, j = n, m
    target_replacement = []
    src_replacement = []
    while i > 0 and j > 0:
        if target_id[i-1] == src_id[j-1]:
            target_replacement.append(i-1)
            src_replacement.append(j-1)
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    # Since backtracking is in reverse order, need to reverse
    target_replacement = target_replacement[::-1]
    src_replacement = src_replacement[::-1]

    print(f'[LCS] replace words: {tokenizer_2.decode(np.array(target_id)[target_replacement])}')
    return target_replacement, src_replacement


def load_pipe_multi(device_map):
    model_path = os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    local_files_only = os.environ.get("FLUX_LOCAL_FILES_ONLY", "0") == "1"
    flux_config = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch.bfloat16,
        "local_files_only": local_files_only
    }

    vae = AutoencoderKL.from_pretrained(
        **flux_config,
        subfolder="vae",
    ).to(device_map['vae'])

    text_encoder = CLIPTextModel.from_pretrained(
        **flux_config,
        subfolder="text_encoder",
    ).to(device_map['text_encoder'])
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        **flux_config,
        subfolder="text_encoder_2",
    ).to(device_map['text_encoder'])

    transformer = FluxTransformer2DModel.from_pretrained(
        **flux_config,
        subfolder="transformer",
    ).to(device_map['transformer'])

    pipe = ParallelFluxPipeline.from_pretrained(
        **flux_config,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        vae=vae
    )
    
    return pipe

def get_tokenized_length(text, pipe):
    """
    Get the tokenized length of a text string using pipe.tokenizer_2
    
    Args:
        text (str): The text to tokenize
        pipe: The pipeline object with tokenizer_2
        
    Returns:
        int: The length of the tokenized text
    """
    tokenized = pipe.tokenizer_2.encode(text, padding=True, max_length=512)
    return len(tokenized)