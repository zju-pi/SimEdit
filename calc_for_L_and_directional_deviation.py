import shutil
import os
import argparse
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

cache_dir = ['src/model/__pycache__','src/util/__pycache__']
for dir in cache_dir: 
    if os.path.exists(dir): shutil.rmtree(dir)

from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

import inspect
from pytorch_lightning import seed_everything
import joblib
from torch.optim import Adam
from PIL import Image

from src.util.utils import show_image, reset_attn, read_image, prompt_embd_aligned_replacement, save_image, load_pipe_multi
from src.util.prompt_runing_multi import save_inversion_latents, run_baseline, run_baseline2, opt, replace
from src.util.visualize import view_self_attn, view_cross_attn, matplotlib_colored_img

# Command line argument parsing
parser = argparse.ArgumentParser(description='Combined LC bounds and cosine similarity evaluation script')
parser.add_argument('--dataset_path', type=str, required=True, help='JSON dataset file path')
parser.add_argument('--prompt_field', type=str, required=True, help='Prompt field to use (e.g., original_prompt, enhanced_source_prompt, structured_enhanced_source)')

parser.add_argument('--latents_dir', type=str, default='latents_combined', help='Directory for storing latents information')
parser.add_argument('--height', type=int, default=512, help='Image height')
parser.add_argument('--width', type=int, default=512, help='Image width')
parser.add_argument('--cuda_device', type=str, default='cuda:0', help='CUDA device, e.g., cuda:0, cuda:1, etc.')
parser.add_argument('--exp_name', type=str, default='', help='Experiment name, used to distinguish different experiments')
parser.add_argument('--annotation_images_dir', type=str, default='../../../data/pie-bench/annotation_images', help='Image folder path')
parser.add_argument('--steps', type=int, default=25, help='Number of inversion steps (default: 25)')
parser.add_argument('--seed', type=int, default=2, help='Random seed (default: 2)')

# Parameters for computing L/C/bound

parser.add_argument('--dt_mode', type=str, default='finite', help='dt v computation mode: finite|jvp|jacobian')
parser.add_argument('--h_fd', type=float, default=2.0, help='Finite difference step size')
parser.add_argument('--power_iters', type=int, default=3, help='Number of power iteration steps')
parser.add_argument('--sample_stride', type=int, default=2, help='Sampling stride')

# Parameters for computing cosine similarity

parser.add_argument('--num_perturb', type=int, default=10, help='Number of perturbation samples (for cosine similarity calculation)')
parser.add_argument('--eps_std', type=float, default=5e-3, help='Perturbation standard deviation (for cosine similarity calculation)')

args = parser.parse_args()

# Set device after parsing arguments
device_map = {
    'vae': args.cuda_device,
    'text_encoder': args.cuda_device,
    'transformer': args.cuda_device
}

## os.environ['http_proxy'] = 'http://127.0.0.1:7890'
## os.environ['https_proxy'] = 'http://127.0.0.1:7890'

print(f"Using CUDA device: {args.cuda_device}")
print(f"Device map: {device_map}")
print(f"Dataset path: {args.dataset_path}")
print(f"Prompt field: {args.prompt_field}")
if args.max_sample is not None:
    print(f"Max sample limit: {args.max_sample}")

# Global variable definitions
DT_MODE = args.dt_mode
H_FD = args.h_fd

# ============================================================================
# Velocity field wrapper v(x,t; P)
# ============================================================================
def velocity_field(pipe, x_latents, t_scalar, prompt_text):
    # prepare prompt embeds
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.get_prompt_embeds(prompt_text)
    pipe.prompt_embeds = prompt_embeds.to(device_map['transformer'])
    pipe.pooled_prompt_embeds = pooled_prompt_embeds.to(device_map['transformer'])
    pipe.text_ids = text_ids.to(device_map['transformer'])

    # network forward (noise prediction)
    # Disable Flash/MemEfficient SDPA so autograd/JVP has a backward implementation
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        noise_pred = pipe.transformer(
            hidden_states=x_latents,
            timestep=t_scalar.to(device_map['transformer']).expand(x_latents.shape[0]) / 1000,
            guidance=pipe.guidance.to(device_map['transformer']),
            pooled_projections=pipe.pooled_prompt_embeds,
            encoder_hidden_states=pipe.prompt_embeds,
            txt_ids=pipe.text_ids,
            img_ids=pipe.latent_image_ids.to(device_map['transformer']),
            return_dict=False,
        )[0]
    return noise_pred

# ============================================================================
# Efficient JVP via autograd.functional
# ============================================================================
from torch.autograd.functional import jvp as autograd_jvp

# ============================================================================
# Estimate Lipschitz constant L
# ============================================================================
def estimate_L_at_point(v_fn, x, t, prompt_text, power_iters=3, eps=1e-8, return_stat="p95"):
    """
    Estimate L(x,t) ≈ ||J_v(x,t)||_2 (spectral norm), supports batch.
    return_stat: "mean"|"p95"|"max"|"both", aggregates over batch.
    When return_stat="both", returns a tuple of (p95, mean).
    mean uses the result from the first random vector, p95 uses the result from power iteration convergence.
    """
    B = x.shape[0]
    x_req = x.detach().requires_grad_(True)

    # per-sample unit vector initialization
    u = torch.randn_like(x_req)
    u = u / (u.flatten(start_dim=1).norm(dim=1, keepdim=True) + eps)

    def f_of_x(z):
        return v_fn(z, t, prompt_text)

    # Compute result from first random vector as "mean" estimate
    if return_stat in ["both", "mean"]:
        _, Ju_random = autograd_jvp(f_of_x, (x.detach(),), (u.detach(),), create_graph=False)
        sigma_random = Ju_random.flatten(start_dim=1).norm(dim=1).float()

    # Power iteration process (for p95 estimate)
    if return_stat in ["both", "p95", "max"]:
        for _ in range(power_iters):
            # 1) Ju via JVP
            _, Ju = autograd_jvp(f_of_x, (x_req,), (u,), create_graph=False)

            # 2) J^T(Ju) via VJP: grad_x <v(x), Ju>
            v_val = f_of_x(x_req)
            JtJu, = torch.autograd.grad(
                outputs=v_val, inputs=x_req, grad_outputs=Ju.detach(),
                retain_graph=False, create_graph=False
            )

            # 3) per-sample normalization
            with torch.no_grad():
                denom = JtJu.flatten(start_dim=1).norm(dim=1, keepdim=True) + eps
                u = (JtJu / denom).detach()

        # final sigma estimate: ||J u|| per sample
        _, Ju_final = autograd_jvp(f_of_x, (x.detach(),), (u.detach(),), create_graph=False)
        sigma_final = Ju_final.flatten(start_dim=1).norm(dim=1).float()

    if return_stat == "both":
        return sigma_final.quantile(0.95).item(), sigma_random.mean().item()
    elif return_stat == "p95":
        return sigma_final.quantile(0.95).item()
    elif return_stat == "max":
        return sigma_final.max().item()
    else:  # mean
        return sigma_random.mean().item()

# ============================================================================
# Estimate constant C
# ============================================================================
def estimate_C_at_point(
    v_fn, x, t, h, prompt_text,
    use_central=True,
    return_batch=False,
    eps=1e-12,
    dt_mode="finite"  # "finite" | "jvp" | "jacobian"
):
    """
    v_fn(x, t, prompt) -> v(x,t;P)
    x: (B, d, ...) or (d, ...)
    t: scalar tensor or broadcastable to batch

    dt_mode:
      - "finite": Use finite difference (default; consistent with original function)
      - "jvp":     Use forward-mode JVP to differentiate w.r.t. t (efficient and stable, requires t's computation graph to be differentiable)
      - "jacobian": Use autograd.functional.jacobian to differentiate w.r.t. t (accurate but slowest, for debugging)
    """
    x_det = x.detach()
    v_t = v_fn(x_det, t, prompt_text).detach()

    # J_v v via JVP (maintain your implementation)
    _, Jv_v = autograd_jvp(lambda z: v_fn(z, t, prompt_text), (x_det,), (v_t,), create_graph=False)

    # Align h's dtype/device
    h_t = h
    if not torch.is_tensor(h_t):
        h_t = torch.tensor(h_t, device=x_det.device, dtype=t.dtype if torch.is_tensor(t) else x_det.dtype)
    else:
        h_t = h_t.to(device=x_det.device, dtype=(t.dtype if torch.is_tensor(t) else x_det.dtype))

    # === Compute ∂_t v ===
    if dt_mode == "finite":
        if use_central and (h_t is not None):
            # Fix: denominator also needs to be divided by 1000 to match the time scale inside velocity_field
            dt_v = (v_fn(x_det, t + h_t, prompt_text) - v_fn(x_det, t - h_t, prompt_text)) / (2.0 * (h_t + eps) / 1000.0)
        else:
            # Fix: denominator also needs to be divided by 1000 to match the time scale inside velocity_field
            dt_v = (v_fn(x_det, t + h_t, prompt_text) - v_fn(x_det, t, prompt_text)) / ((h_t + eps) / 1000.0)

    elif dt_mode == "jvp":
        # Use JVP to differentiate w.r.t. t: dv/dt = J_{v wrt t} · 1
        t_in = t if torch.is_tensor(t) else torch.tensor(t, device=x_det.device, dtype=x_det.dtype)
        # Note: here we fix x and only build graph for t
        _, dt_v = autograd_jvp(lambda tt: v_fn(x_det, tt, prompt_text),
                               (t_in.detach(),),
                               (torch.ones_like(t_in),),
                               create_graph=False)

    elif dt_mode == "jacobian":
        # Directly compute Jacobian w.r.t. t, then extract dv/dt; slowest, for debugging
        from torch.autograd.functional import jacobian
        t_in = t if torch.is_tensor(t) else torch.tensor(t, device=x_det.device, dtype=x_det.dtype)
        dt_v = jacobian(lambda tt: v_fn(x_det, tt, prompt_text), t_in.detach())
        # For scalar t: dt_v shape == v shape; for batch t: returns outer dimension matching t shape, higher cost

    else:
        raise ValueError(f"Unknown dt_mode: {dt_mode}")

    term = Jv_v + dt_v

    if term.dim() == 1:
        C_val = 0.5 * term.norm()
        return C_val.item()
    else:
        C_vec = 0.5 * term.flatten(1).norm(dim=1)
        return C_vec if return_batch else C_vec.mean().item()

# ============================================================================
# Gather L/C values over trajectory
# ============================================================================
def gather_LC_over_traj(pipe, prompt_text, latents_dir, steps, sample_stride=3, power_iters=3, dt_mode="finite"):
    L_vals = []
    L_vals_mean = []  # Store mean value for each timestep
    C_vals = []
    h_vals = []
    for t_idx in range(1, steps + 1, sample_stride):
        x_t = torch.load(os.path.join(latents_dir, f'latents_{t_idx}.pt'), map_location=device_map['vae'])
        t_cur = torch.tensor([pipe.scheduler.timesteps[steps - t_idx]]).to(x_t.device)
        if t_idx == 1:
            t_prev = torch.tensor([0.0], device=x_t.device)
        else:
            t_prev = torch.tensor([pipe.scheduler.timesteps[steps - t_idx + 1]], device=x_t.device)
        # integrator step for Euler update
        h = (t_prev - t_cur).abs() / 1000.0
        # robust step for dt v finite difference (independent of integrator h)
        h_fd_tensor = torch.tensor(H_FD, device=x_t.device, dtype=t_cur.dtype)
        L_pt, L_pt_mean = estimate_L_at_point(lambda z, t, p: velocity_field(pipe, z, t, p), x_t, t_cur, prompt_text, power_iters=power_iters, return_stat="both")
        C_pt = estimate_C_at_point(lambda z, t, p: velocity_field(pipe, z, t, p), x_t, t_cur, h_fd_tensor, prompt_text, use_central=True, return_batch=False, eps=1e-12, dt_mode=dt_mode)
        L_vals.append(L_pt)
        L_vals_mean.append(L_pt_mean)
        C_vals.append(C_pt)
        h_vals.append(h.item())
        print(f't_idx={t_idx}: L~{L_pt:.4e}, L_mean~{L_pt_mean:.4e}, C~{C_pt:.4e}, h_int={h.item():.4e}, h_fd={float(h_fd_tensor):.6f}')
    L_emp = float(np.percentile(np.array(L_vals), 95))
    L_mean = float(np.mean(L_vals_mean))  # Double average (averaged over both spatial and temporal dimensions)
    C_emp = float(np.percentile(np.array(C_vals), 95))
    h_mean = float(np.mean(h_vals))
    h_max = float(np.max(h_vals))
    # total time horizon using scheduler endpoints (more faithful than steps*h)
    try:
        t0 = float(pipe.scheduler.timesteps[0])
        tT = float(pipe.scheduler.timesteps[-1])
        T_total = (t0 - tT) / 1000.0
    except Exception:
        T_total = steps * h_mean
    return {
        'L_vals': L_vals,
        'L_vals_mean': L_vals_mean,  # Mean value for each timestep
        'C_vals': C_vals,
        'h_vals': h_vals,
        'L_emp': L_emp,
        'L_mean': L_mean,  # Double average
        'C_emp': C_emp,
        'h_mean': h_mean,
        'h_max': h_max,
        'T_total': T_total,
    }

# ============================================================================
# Compute Euler two-pass bound
# ============================================================================
def compute_euler_two_pass_bound(L_emp, C_emp, T_total, h):
    # \widehat{B}(P_src)=2*(C/L)*(exp(L*T)-1)*h
    if L_emp <= 0:
        return float('inf')
    K = 2.0 * (C_emp / L_emp) * (np.exp(L_emp * T_total) - 1.0)
    return float(K * h)

# ============================================================================
# Reconstruction error
# ============================================================================
def reconstruction_error(pipe, prompt_text, latents_dir, steps, base_cfg, image):
    # reuse stored x_T from inversion_dir and reconstruct with same P_src
    start_latents = torch.load(os.path.join(latents_dir, f'latents_{steps}.pt'), map_location=device_map['vae'])
    recon = run_baseline2(pipe, prompt_text, start_latents=start_latents, start_steps=0, guidance_scale=1, **base_cfg, emphasize_scale=1)
    # x0 image
    x0_img = image
    # to tensor
    def to_tensor(img):
        if isinstance(img, torch.Tensor):
            return img
        return torch.from_numpy(np.array(img)) if not isinstance(img, np.ndarray) else torch.from_numpy(img)
    rec_img_t = to_tensor(recon["images"][0]).float()
    x0_img_t = to_tensor(x0_img).float()
    # align shapes
    if rec_img_t.shape != x0_img_t.shape:
        min_h = min(rec_img_t.shape[-2], x0_img_t.shape[-2])
        min_w = min(rec_img_t.shape[-1], x0_img_t.shape[-1])
        rec_img_t = rec_img_t[..., :min_h, :min_w]
        x0_img_t = x0_img_t[..., :min_h, :min_w]
    # ensure same device
    rec_img_t = rec_img_t.to(x0_img_t.device)
    err = rec_img_t - x0_img_t
    mse = (err ** 2).mean().item()
    l2 = err.flatten().norm().item() / np.sqrt(err.numel())
    return {"mse": mse, "l2": l2, "rec_img": recon["images"][0]}

# ============================================================================
# Perturb and sample for cosine similarity calculation
# ============================================================================
@torch.no_grad()
def perturb_and_sample(pipe, xt, t_cur, t_prev, source_prompt, num_samples=8, eps_std=5e-3):
    """
    Add perturbation to latent, compute cosine similarity of noise prediction before and after perturbation.
    Returns: (xtm1_var, gra1_var, cos1_mean)
    """
    xtm1_list = []
    gra1_list = []
    cos1_list = []
    # Ensure xt is on the correct device
    xt = xt.to(device_map['transformer'])
    
    # Get prompt embeds
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.get_prompt_embeds(source_prompt)
    pipe.prompt_embeds = prompt_embeds.to(device_map['transformer'])
    pipe.pooled_prompt_embeds = pooled_prompt_embeds.to(device_map['transformer'])
    pipe.text_ids = text_ids.to(device_map['transformer'])

    noise_pred0 = pipe.transformer(
            hidden_states=xt,
            timestep=t_cur.to(device_map['transformer']).expand(xt.shape[0]) / 1000,
            guidance=pipe.guidance.to(device_map['transformer']),
            pooled_projections=pipe.pooled_prompt_embeds,
            encoder_hidden_states=pipe.prompt_embeds,
            txt_ids=pipe.text_ids,
            img_ids=pipe.latent_image_ids.to(device_map['transformer']),
            return_dict=False,
        )[0]

    for _ in range(num_samples):
        delta = torch.randn_like(xt) * eps_std
        xt_delta = xt + delta
        xt_delta = xt_delta.to(device_map['transformer'])
        noise_pred = pipe.transformer(
            hidden_states=xt_delta,
            timestep=t_cur.to(device_map['transformer']).expand(xt.shape[0]) / 1000,
            guidance=pipe.guidance.to(device_map['transformer']),
            pooled_projections=pipe.pooled_prompt_embeds,
            encoder_hidden_states=pipe.prompt_embeds,
            txt_ids=pipe.text_ids,
            img_ids=pipe.latent_image_ids.to(device_map['transformer']),
            return_dict=False,
        )[0]
        xt_delta = xt_delta.to(torch.float32)
        xtm1 = xt_delta + (t_prev - t_cur) / 1000. * noise_pred
        xtm1 = xtm1.to(noise_pred.dtype)
        cos = 1 - torch.nn.functional.cosine_similarity(noise_pred.flatten().double(), noise_pred0.flatten().double(), dim=0)
        xtm1_list.append(xtm1.cpu())
        gra1_list.append(noise_pred.cpu())
        cos1_list.append(cos.cpu().double())

    xtm1_stack = torch.stack(xtm1_list, dim=0)  # [num_samples, B, C, H, W]
    xtm1_var = torch.var(xtm1_stack.float(), dim=0).mean().item()  # scalar
    gra1_stack = torch.stack(gra1_list, dim=0)  # [num_samples, B, C, H, W]
    gra1_var = torch.var(gra1_stack.float(), dim=0).mean().item()  # scalar
    cos1_mean = torch.stack(cos1_list).mean().item()
    
    return xtm1_var, gra1_var, cos1_mean

# ============================================================================
# Inversion robustness test (cosine similarity over trajectory)
# ============================================================================
def inversion_robustness_test(pipe,
        latents_save_dir,
        source_prompt,
        device_map,
        steps,
        width, height,
        guidance_scale=1,
        first_order=False,
        num_perturb=10,
        eps_std=5e-3,
        **args
        ):
    """
    Iterate through timesteps, compute mean cosine similarity for each timestep.
    Returns: list of [(t_idx, cos_score), ...]
    """
    robustness_scores = []
   
    for t_idx in tqdm(range(1, steps+1), desc=f"Robustness ({source_prompt[:20]}...)"):
        xt = torch.load(os.path.join(latents_save_dir, f'latents_{t_idx}.pt'), map_location=device_map['vae'])
        t_cur = torch.tensor([pipe.scheduler.timesteps[steps-t_idx]]).to(device_map['transformer'])
        if t_idx == 1:
            t_prev = torch.tensor([0.0]).to(device_map['transformer'])
        else:
            t_prev = torch.tensor([pipe.scheduler.timesteps[steps-t_idx+1]]).to(device_map['transformer'])
        var_score, gra_score, cos_score = perturb_and_sample(
            pipe, xt, t_cur, t_prev, source_prompt=source_prompt, 
            num_samples=num_perturb, eps_std=eps_std
        )
        robustness_scores.append((t_idx, cos_score))
    return robustness_scores

# ============================================================================
# Combined Evaluator Class
# ============================================================================
class CombinedEvaluator:
    def __init__(self, dataset_path, prompt_field, device_map, annotation_images_dir, pipe):
        self.dataset_path = dataset_path
        self.prompt_field = prompt_field
        self.device_map = device_map
        self.annotation_images_dir = annotation_images_dir
        self.pipe = pipe
        self.HEIGHT = args.height
        self.WIDTH = args.width
        self.STEPS = args.steps
        self.SEED = args.seed
        
    def load_dataset(self):
        """Load the JSON dataset (format like 100words_mapping_4.1.json)"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("JSON file must contain a dictionary at the top level.")
        
        return data
    
    def evaluate_sample(self, item_name, item_data, output_dir):
        """Evaluate LC bounds and cosine similarity for a single sample"""
        # Get prompt from specified field
        if self.prompt_field not in item_data:
            print(f"Warning: Field '{self.prompt_field}' not found in item {item_name}. Skipping.")
            return None
        
        source_prompt = item_data[self.prompt_field]
        
        # Check if prompt is valid (not empty)
        if not source_prompt or not isinstance(source_prompt, str) or not source_prompt.strip():
            print(f"Warning: Invalid prompt in field '{self.prompt_field}' for item {item_name}. Skipping.")
            return None
        
        image_path = item_data.get("image_path", "")
        
        if not image_path:
            print(f"Warning: image_path not found in item {item_name}. Skipping.")
            return None
        
        # Create directories
        item_dir = os.path.join(output_dir, item_name)
        os.makedirs(item_dir, exist_ok=True)
        
        # Check if original image exists
        original_image_path = os.path.join(self.annotation_images_dir, image_path)
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image not found at {original_image_path}")
            return None
            
        # Read image
        image = read_image(original_image_path, self.HEIGHT)
        actual_width = image.shape[2]
        timesteps, steps = self.pipe.prepare_timesteps(self.STEPS, (actual_width // 16) * (self.HEIGHT // 16), self.device_map['vae'])

        # Prepare configuration
        base_config = {
            "device_map": self.device_map,
            "steps": self.STEPS,
            "first_order": False,
            "width": actual_width,
            "height": self.HEIGHT,
            "seed": self.SEED
        }

        # Create latents directory (sanitize prompt for filesystem)
        safe_prompt = source_prompt[:50].replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        latents_save_dir = os.path.join(item_dir, f"latents_{safe_prompt}")
        
        try:
            # Save inversion latents
            source_x0, source_xT, source_img = save_inversion_latents(
                self.pipe, source_prompt, timesteps, latents_save_dir, image, **base_config
            )
            
            # Calculate L/C statistics
            lc_stats = gather_LC_over_traj(
                self.pipe, source_prompt, latents_save_dir, steps, 
                sample_stride=args.sample_stride, 
                power_iters=args.power_iters, 
                dt_mode=DT_MODE
            )
            
            # Calculate bound (using h_mean)
            bound_val = compute_euler_two_pass_bound(
                lc_stats['L_emp'], lc_stats['C_emp'], 
                lc_stats['T_total'], lc_stats['h_mean']
            )
            
            # Calculate reconstruction error
            rec_metrics = reconstruction_error(
                self.pipe, source_prompt, latents_save_dir, steps, base_config, image
            )
            
            # Calculate cosine similarity robustness scores
            # Exclude parameters that are explicitly passed to avoid duplicate arguments:
            # - device_map, steps, width, height are passed explicitly (use actual values, not base_config)
            # - first_order and seed can be passed via **config_for_robustness if needed
            excluded_params = {'device_map', 'steps', 'width', 'height'}
            config_for_robustness = {k: v for k, v in base_config.items() if k not in excluded_params}
            robustness_scores = inversion_robustness_test(
                self.pipe,
                latents_save_dir=latents_save_dir,
                source_prompt=source_prompt,
                device_map=self.device_map,
                steps=steps,  # Use actual steps from prepare_timesteps, not self.STEPS
                width=actual_width,
                height=self.HEIGHT,
                num_perturb=args.num_perturb,
                eps_std=args.eps_std,
                **config_for_robustness  # Contains: first_order, seed (if present)
            )
            
            # Calculate mean cosine similarity across all timesteps
            cos_scores_list = [cos_score for _, cos_score in robustness_scores if cos_score is not None and not (np.isnan(cos_score) or np.isinf(cos_score))]
            cos_mean = float(np.mean(cos_scores_list)) if cos_scores_list else None
            cos_std = float(np.std(cos_scores_list)) if cos_scores_list and len(cos_scores_list) > 1 else None
            
            # Prepare return value
            result = {
                'lc_stats': lc_stats,
                'bound_val': bound_val,
                'rec_metrics': rec_metrics,
                'robustness_scores': robustness_scores,
                'cos_mean': cos_mean,
                'cos_std': cos_std,
                'actual_width': actual_width,
                'image_path': image_path,
                'item_name': item_name,
                'source_prompt': source_prompt
            }
            
            # Delete latents directory to save disk space after all computations are done
            # Note: latents_save_dir is a subdirectory under item_dir, and CSV files are saved
            # in exp_latents_dir (parent of item_dir), so deleting latents_save_dir is safe
            if os.path.exists(latents_save_dir):
                try:
                    # Double check: ensure we're only deleting the latents subdirectory, not the whole exp_latents_dir
                    if 'latents_' in os.path.basename(latents_save_dir):
                        shutil.rmtree(latents_save_dir)
                        print(f"Deleted latents directory: {latents_save_dir}")
                    else:
                        print(f"Warning: Skipping deletion - directory name doesn't match expected pattern: {latents_save_dir}")
                except Exception as cleanup_error:
                    print(f"Warning: Failed to delete latents directory {latents_save_dir}: {cleanup_error}")
            
            return result
            
        except Exception as e:
            print(f"Exception occurred for {latents_save_dir}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up latents directory on error as well
            if os.path.exists(latents_save_dir):
                try:
                    shutil.rmtree(latents_save_dir)
                    print(f"Cleaned up latents directory after error: {latents_save_dir}")
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up latents directory {latents_save_dir}: {cleanup_error}")
            return None

# ============================================================================
# Main function
# ============================================================================
def main():
    # Load model
    pipe = load_pipe_multi(device_map)
    pipe.transformer.eval()
    print("Model loaded successfully!")

    # Create experiment-level isolated directory
    exp_suffix = f"_{args.exp_name}" if args.exp_name else ""
    exp_dir = f"exp_{args.cuda_device.replace(':', '')}_{DT_MODE}_h{H_FD}_p{args.power_iters}_s{args.sample_stride}_{args.height}x{args.width}_{args.prompt_field}{exp_suffix}"
    exp_latents_dir = os.path.join(args.latents_dir, exp_dir)
    os.makedirs(exp_latents_dir, exist_ok=True)

    # Create evaluator
    evaluator = CombinedEvaluator(
        args.dataset_path,
        args.prompt_field,
        device_map,
        args.annotation_images_dir,
        pipe
    )
    
    # Load dataset
    dataset = evaluator.load_dataset()
    total_items = len(dataset)
    
    # Convert to list for indexing
    dataset_items = list(dataset.items())
    
    # Validate start_index
    if args.start_index < 0:
        args.start_index = 0
        print(f"Warning: start_index < 0, setting to 0")
    if args.start_index >= total_items:
        print(f"Error: start_index ({args.start_index}) >= total_items ({total_items})")
        return
    
    # Calculate end index based on start_index and max_sample
    if args.max_sample is not None and args.max_sample > 0:
        end_index = min(args.start_index + args.max_sample, total_items)
        dataset_items = dataset_items[args.start_index:end_index]
        dataset = dict(dataset_items)
        print(f"Loaded dataset with {total_items} items, processing items {args.start_index} to {end_index-1} (total {len(dataset)} items)")
    else:
        dataset_items = dataset_items[args.start_index:]
        dataset = dict(dataset_items)
        print(f"Loaded dataset with {total_items} items, processing from index {args.start_index} to end (total {len(dataset)} items)")
    
    results = []
    all_detailed_results = []
    all_trajectory_cos_results = []
    
    processed_count = 0
    
    # Iterate through all items in the dataset
    for item_name, item_data in tqdm(dataset.items(), desc="Processing samples"):
        # Calculate global index (for display purposes)
        global_index = args.start_index + processed_count
        print(f"\n{'='*60}")
        print(f"Processing item {item_name} (global index: {global_index}, local: {processed_count + 1}/{len(dataset)})")
        print(f"{'='*60}")
        
        # Evaluate sample
        result = evaluator.evaluate_sample(item_name, item_data, exp_latents_dir)
        
        if result:
            lc_stats = result['lc_stats']
            bound_val = result['bound_val']
            rec_metrics = result['rec_metrics']
            robustness_scores = result['robustness_scores']
            cos_mean = result['cos_mean']
            cos_std = result['cos_std']
            actual_width = result['actual_width']
            image_path = result['image_path']
            source_prompt = result['source_prompt']
            
            # Prepare summary data
            summary_data = {
                'item_name': item_name,
                'image_path': image_path,
                'prompt_field': args.prompt_field,
                'source_prompt': source_prompt[:200],  # Truncate for CSV
                'L_emp': lc_stats['L_emp'],
                'L_mean': lc_stats['L_mean'],
                'C_emp': lc_stats['C_emp'],
                'h_mean': lc_stats['h_mean'],
                'h_max': lc_stats['h_max'],
                'T_total': lc_stats['T_total'],
                'bound_mean': bound_val,
                'bound_max': compute_euler_two_pass_bound(
                    lc_stats['L_emp'], lc_stats['C_emp'], 
                    lc_stats['T_total'], lc_stats['h_max']
                ),
                'recon_l2': rec_metrics['l2'],
                'recon_mse': rec_metrics['mse'],
                'cos_mean': cos_mean,
                'cos_std': cos_std,
                'dt_mode': DT_MODE,
                'h_fd': H_FD,
                'power_iters': args.power_iters,
                'sample_stride': args.sample_stride,
                'height': args.height,
                'width': actual_width,
                'num_perturb': args.num_perturb,
                'eps_std': args.eps_std,
                'cuda_device': args.cuda_device,
                'exp_name': args.exp_name
            }
            
            results.append(summary_data)
            
            # Prepare detailed trajectory data (each step's L and C)
            trajectory_data = []
            # Calculate the actual number of steps based on sample_stride
            num_steps = len(lc_stats['L_vals'])
            for idx, (L_val, L_val_mean, C_val, h_val) in enumerate(zip(
                lc_stats['L_vals'],
                lc_stats['L_vals_mean'],
                lc_stats['C_vals'],
                lc_stats['h_vals']
            )):
                t_idx = 1 + idx * args.sample_stride
                trajectory_data.append({
                    'item_name': item_name,
                    'image_path': image_path,
                    'prompt_field': args.prompt_field,
                    't_idx': t_idx,
                    'L_value': L_val,
                    'L_value_mean': L_val_mean,
                    'C_value': C_val,
                    'h_value': h_val,
                    'dt_mode': DT_MODE,
                    'h_fd': H_FD,
                    'power_iters': args.power_iters,
                    'sample_stride': args.sample_stride,
                })
            
            all_detailed_results.extend(trajectory_data)
            
            # Prepare cosine similarity trajectory data
            cos_trajectory_data = []
            for t_idx, cos_score in robustness_scores:
                if cos_score is not None:
                    entry = {
                        'item_name': item_name,
                        'image_path': image_path,
                        'prompt_field': args.prompt_field,
                        't_idx': t_idx,
                        'cos_score': cos_score,
                        'num_perturb': args.num_perturb,
                        'eps_std': args.eps_std,
                    }
                    cos_trajectory_data.append(entry)
                    all_trajectory_cos_results.append(entry)
            
            # Save individual sample results to its own folder
            item_dir = os.path.join(exp_latents_dir, item_name)
            os.makedirs(item_dir, exist_ok=True)
            
            # Save summary for this sample
            sample_summary_df = pd.DataFrame([summary_data])
            sample_summary_df.to_csv(os.path.join(item_dir, f'{item_name}_summary.csv'), index=False)
            
            # Save LC trajectory for this sample
            if trajectory_data:
                sample_trajectory_df = pd.DataFrame(trajectory_data)
                sample_trajectory_df.to_csv(os.path.join(item_dir, f'{item_name}_LC_trajectory.csv'), index=False)
            
            # Save cosine similarity trajectory for this sample
            if cos_trajectory_data:
                sample_cos_df = pd.DataFrame(cos_trajectory_data)
                sample_cos_df.to_csv(os.path.join(item_dir, f'{item_name}_cosine_similarity_trajectory.csv'), index=False)
            
            print(f"Saved individual results for {item_name} to {item_dir}/")
            
            print(f"Results: L_emp={lc_stats['L_emp']:.4e}, L_mean={lc_stats['L_mean']:.4e}, "
                  f"C_emp={lc_stats['C_emp']:.4e}, Bound={bound_val:.4e}, "
                  f"Recon_L2={rec_metrics['l2']:.4e}, Cos_Mean={cos_mean:.4e}")
        
        processed_count += 1
    
    # Save final results (always save if we have any results, even if processing was incomplete)
    # Note: If start_index > 0, we need to load existing results and merge
    if results:
        final_results_path = os.path.join(exp_latents_dir, 'final_combined_results.csv')
        
        # If start_index > 0, try to load and merge existing results
        if args.start_index > 0 and os.path.exists(final_results_path):
            try:
                existing_df = pd.read_csv(final_results_path)
                new_df = pd.DataFrame(results)
                # Merge: remove any existing items that overlap with new results, then append
                # Match by item_name
                existing_df = existing_df[~existing_df['item_name'].isin(new_df['item_name'])]
                results_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"\nMerged with existing results: {len(existing_df)} existing + {len(new_df)} new = {len(results_df)} total")
            except Exception as e:
                print(f"\nWarning: Could not merge with existing results: {e}")
                print(f"Creating new results file instead")
                results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame(results)
        
        results_df.to_csv(final_results_path, index=False)
        print(f"\nSaved final results: {final_results_path} (total {len(results_df)} items)")
        
        # Save combined detailed LC trajectory results
        # Note: For trajectory files, we append new results if start_index > 0
        if all_detailed_results:
            lc_trajectory_path = os.path.join(exp_latents_dir, 'all_detailed_LC_trajectory.csv')
            new_trajectory_df = pd.DataFrame(all_detailed_results)
            
            if args.start_index > 0 and os.path.exists(lc_trajectory_path):
                try:
                    existing_trajectory_df = pd.read_csv(lc_trajectory_path)
                    # Remove overlapping items by item_name
                    existing_trajectory_df = existing_trajectory_df[~existing_trajectory_df['item_name'].isin(new_trajectory_df['item_name'])]
                    combined_trajectory_df = pd.concat([existing_trajectory_df, new_trajectory_df], ignore_index=True)
                    combined_trajectory_df.to_csv(lc_trajectory_path, index=False)
                    print(f"Saved detailed LC trajectory: {lc_trajectory_path} (merged, total {len(combined_trajectory_df)} rows)")
                except Exception as e:
                    print(f"Warning: Could not merge LC trajectory: {e}, creating new file")
                    new_trajectory_df.to_csv(lc_trajectory_path, index=False)
                    print(f"Saved detailed LC trajectory: {lc_trajectory_path} ({len(new_trajectory_df)} rows)")
            else:
                new_trajectory_df.to_csv(lc_trajectory_path, index=False)
                print(f"Saved detailed LC trajectory: {lc_trajectory_path} ({len(new_trajectory_df)} rows)")
        
        # Save cosine similarity trajectory results
        if all_trajectory_cos_results:
            cos_trajectory_path = os.path.join(exp_latents_dir, 'all_cosine_similarity_trajectory.csv')
            new_cos_df = pd.DataFrame(all_trajectory_cos_results)
            
            if args.start_index > 0 and os.path.exists(cos_trajectory_path):
                try:
                    existing_cos_df = pd.read_csv(cos_trajectory_path)
                    # Remove overlapping items by item_name
                    existing_cos_df = existing_cos_df[~existing_cos_df['item_name'].isin(new_cos_df['item_name'])]
                    combined_cos_df = pd.concat([existing_cos_df, new_cos_df], ignore_index=True)
                    combined_cos_df.to_csv(cos_trajectory_path, index=False)
                    print(f"Saved cosine similarity trajectory: {cos_trajectory_path} (merged, total {len(combined_cos_df)} rows)")
                except Exception as e:
                    print(f"Warning: Could not merge cosine trajectory: {e}, creating new file")
                    new_cos_df.to_csv(cos_trajectory_path, index=False)
                    print(f"Saved cosine similarity trajectory: {cos_trajectory_path} ({len(new_cos_df)} rows)")
            else:
                new_cos_df.to_csv(cos_trajectory_path, index=False)
                print(f"Saved cosine similarity trajectory: {cos_trajectory_path} ({len(new_cos_df)} rows)")
        
        # Calculate and save summary statistics (based on merged results_df)
        summary_stats = [{
            'prompt_field': args.prompt_field,
            'count': len(results_df),
            'avg_L_emp': results_df['L_emp'].mean(),
            'avg_L_mean': results_df['L_mean'].mean(),
            'avg_C_emp': results_df['C_emp'].mean(),
            'avg_bound_mean': results_df['bound_mean'].mean(),
            'avg_recon_l2': results_df['recon_l2'].mean(),
            'avg_cos_mean': results_df['cos_mean'].mean(),
            'std_L_emp': results_df['L_emp'].std(),
            'std_L_mean': results_df['L_mean'].std(),
            'std_C_emp': results_df['C_emp'].std(),
            'std_bound_mean': results_df['bound_mean'].std(),
            'std_recon_l2': results_df['recon_l2'].std(),
            'std_cos_mean': results_df['cos_mean'].std()
        }]
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(exp_latents_dir, 'combined_summary.csv'), index=False)
        print(f"Saved summary statistics: {os.path.join(exp_latents_dir, 'combined_summary.csv')}")
        
        print(f"\n{'='*60}")
        print("COMBINED EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total items processed: {processed_count}")
        print(f"Total results saved: {len(results)}")
        print(f"Prompt field used: {args.prompt_field}")
        print(f"Results saved to: {os.path.join(exp_latents_dir, 'final_combined_results.csv')}")
        print(f"Summary saved to: {os.path.join(exp_latents_dir, 'combined_summary.csv')}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("WARNING: No results were saved!")
        print(f"{'='*60}")
        print(f"Processed {processed_count} items but no successful results.")
        print("This could mean:")
        print("  1. All samples failed processing (check error messages above)")
        print("  2. Script was interrupted before completion")
        print("  3. Samples were skipped due to missing images or invalid prompts")
        print(f"{'='*60}")
    
    print("\nAll combined analysis completed!")

if __name__ == "__main__":
    main()

