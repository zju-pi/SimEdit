import os
import torch
from torch.optim import Adam
from pytorch_lightning import seed_everything

from src.model.attention import StoreAttention, AttentionReplacement
from src.util.utils import show_image, reset_attn, read_image, prompt_embd_aligned_replacement, save_image, prompt_embd_aligned_replacement_advanced

single_location = f"single_transformer_blocks.[{','.join([str(i) for i in range(38)])}].attn"
multi_location = f"transformer_blocks.[{','.join([str(i) for i in range(19)])}].attn"
replacement_location = f"single_transformer_blocks.[{','.join([str(i) for i in range(20,38)])}].attn"

def reg_save_mid_step(self, save=True):
    def step(latents,i,t_cur,t_prev,first_order=True,inverse=False,prev_latents=None,t_cur_=None):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if t_cur_ is None:
            t_cur_ = t_cur
        timestep = t_cur_.expand(latents.shape[0]).to(latents.dtype)

        latents = self.get_latents_input(latents,t_cur_)
        if i == 0: print(f'Input latents shape: {latents.shape}')
        if i == 0: print(f'Input text embeds shape: {self.prompt_embeds.shape}')

        # register timestep
        for location in self.time_location:
            self.register_time(location,(t_prev if inverse else t_cur).item(),i)

            # reg step
            self.register(location,lambda model,arr_stack,where: setattr(model,'fo',0))

        timestep = timestep.to(self.device_map['transformer'])
        latents = latents.to(self.device_map['transformer'])
        self.prompt_embeds = self.prompt_embeds.to(self.device_map['transformer'])

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=self.guidance,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=self.latent_image_ids,
            return_dict=False,
        )[0]

        noise_pred = noise_pred.to(self.device_map['vae'])
        latents = latents.to(self.device_map['vae'])

        if first_order:
            latents_mid = latents + (t_prev - t_cur) * noise_pred / 2 / 1000 
            timestep_mid = (t_cur + (t_prev - t_cur) / 2).expand(latents.shape[0]).to(latents.dtype)

            # reg step
            for location in self.time_location:
                self.register(location,lambda model,arr_stack,where: setattr(model,'fo',1))

            latents_mid = latents_mid.to(self.device_map['transformer'])
            timestep_mid = timestep_mid.to(self.device_map['transformer'])

            noise_pred_mid = self.transformer(
                hidden_states=latents_mid,
                timestep=timestep_mid / 1000,
                guidance=self.guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=self.latent_image_ids,
                return_dict=False,
            )[0]

            noise_pred_mid = noise_pred_mid.to(self.device_map['vae'])
            latents_mid = latents_mid.to(self.device_map['vae'])

            # compute the previous noisy sample x_t -> x_t-1
            first_order_noise = (noise_pred_mid - noise_pred) * 2 * 1000 / (t_prev - t_cur)

        self.prompt_embeds = self.prompt_embeds.to(self.device_map['text_encoder'])
        if prev_latents is None:
            latents = latents.to(torch.float32)
            latents = latents + (t_prev - t_cur) / 1000. * noise_pred + (0.5 * ((t_prev - t_cur) / 1000) ** 2 * first_order_noise if first_order else 0)
            latents = latents.to(noise_pred.dtype)
        else:
            latents = latents.to(torch.float32)
            prev_latents = prev_latents.to(torch.float32)
            latents = prev_latents + (t_prev - t_cur) / 1000. * noise_pred + (0.5 * ((t_prev - t_cur) / 1000) ** 2 * first_order_noise if first_order else 0)
            latents = latents.to(noise_pred.dtype)

        ### Save latents
        if save:
            os.makedirs(self.path,exist_ok=True)
            torch.save(latents.cpu(),os.path.join(self.path,f'latents_{i+1}.pt'))

        return latents
    self.step = step

def save_inversion_latents(pipe,prompt,timesteps,save_path,image,device_map,steps,first_order,guidance_scale=1,inversion_method="fixed_point",store_attn=True,**args):
    if save_path is not None: reg_save_mid_step(pipe)
    reset_attn(pipe)
    
    attn = StoreAttention(timesteps, store_attn = store_attn, attn_map_path = 'attn_map', save_type = 'avg_t' ,replacement_path=save_path)
    pipe.register_attn(replacement_location,attn)
    
    out = pipe.invert(
        device_map=device_map,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        first_order=first_order,
        image=image,
        path=save_path,
        inversion_method=inversion_method
    )
    if store_attn:
        attn.flush()
    reg_save_mid_step(pipe,False)
    x0 = out["start_latents"]
    xT = out["latents"]
    img = out['images']

    print((torch.load(os.path.join(save_path,f'latents_{steps}.pt')).cpu()-xT.cpu()).mean())
    if save_path is not None: torch.save(x0,os.path.join(save_path,f'latents_0.pt'))
    
    return x0, xT, img

def run_baseline(pipe,prompt,device_map,steps,width,height,start_latents=None, start_steps = 0, first_order=False,seed=2,guidance_scale=2,emphasize_scale=1,spec_id=None):
    reg_save_mid_step(pipe,False)
    reset_attn(pipe)
    
    def find_first_padding(txt_id):
        idx = 511
        while idx > 0:
            if txt_id[idx] != 0:
                idx += 1
                break
            idx -= 1
        return idx
    txt_id = pipe.tokenizer_2.encode(prompt,padding='max_length',max_length=512)
    idx = find_first_padding(txt_id)
    attn =AttentionReplacement(inject_timestep=range(0,0), max_token_id=idx, emphasize_scale=emphasize_scale, mix_ratios=(0, 0, 1), spec_replace_id=spec_id)
    reset_attn(pipe)
    pipe.register_attn(replacement_location,attn)
    # seed_everything(seed)
    return pipe(
        device_map=device_map,
        prompt=prompt,
        guidance_scale=guidance_scale,
        latents=start_latents,
        start_steps = start_steps,
        height=height,
        width=width,
        num_inference_steps=steps,
        first_order=first_order,
    )

def run_baseline_save_attn(pipe,prompt,device_map,steps,width,height,start_latents=None, start_steps = 0, first_order=False,seed=2,guidance_scale=2,emphasize_scale=1,spec_id=None):
    reset_attn(pipe)
    
    attn = StoreAttention(inject_timestep=range(0,0), store_attn = True, store_qkv = False, attn_map_path = 'attn_map', save_type = 'avg_t')
    pipe.register_attn(replacement_location,attn)
    # seed_everything(seed)
    return pipe(
        device_map=device_map,
        prompt=prompt,
        guidance_scale=guidance_scale,
        latents=start_latents,
        start_steps = start_steps,
        height=height,
        width=width,
        num_inference_steps=steps,
        first_order=first_order,
    )

def run_baseline2(pipe,prompt,device_map,steps,width,height,start_latents=None, start_steps = 0, first_order=False,seed=2,guidance_scale=2,emphasize_scale=1):
    reg_save_mid_step(pipe,False)
    reset_attn(pipe)
    seed_everything(seed)
    # seed_everything(seed)
    return pipe(
        device_map=device_map,
        prompt=prompt,
        guidance_scale=guidance_scale,
        latents=start_latents,
        start_steps = start_steps,
        height=height,
        width=width,
        num_inference_steps=steps,
        first_order=first_order,
    )


def replace_enhance(pipe,
            latents_save_dir,
            source_prompt,
            target_prompt,
            timesteps,
            device_map,
            steps,
            width, height,
            guidance_scale=1,
            first_order=False,
            emphasize_scale=2,
            **args
            ):
    
    def find_first_padding(txt_id):
        idx = 511
        while idx > 0:
            if txt_id[idx] != 0:
                idx += 1
                break
            idx -= 1
        return idx
    txt_id = pipe.tokenizer_2.encode(target_prompt,padding='max_length',max_length=512)
    idx = find_first_padding(txt_id)
    spec_replace_id=prompt_embd_aligned_replacement_advanced(source_prompt,target_prompt,pipe.tokenizer_2)
    attn = AttentionReplacement(timesteps, replacement_path=latents_save_dir, max_token_id=idx, \
        emphasize_scale=emphasize_scale, mix_ratios=(0, 0, 1), spec_replace_id=spec_replace_id)

    start_latents = torch.load(os.path.join(latents_save_dir,f'latents_{steps}.pt'),device_map['vae'])

    reset_attn(pipe)
    pipe.register_attn(replacement_location,attn)

    return pipe(
        device_map=device_map,
        prompt=target_prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=steps,
        latents=start_latents,
        first_order=first_order,
    )



def save_mid_latents(pipe,prompt,timesteps,start_latents,save_path,device_map,seed,steps,width,height,guidance_scale,first_order,**args):
    reg_save_mid_step(pipe)
    reset_attn(pipe)

    attn = StoreAttention(timesteps,store_attn = False, attn_map_path = 'attn_map', save_type = 'avg_t' ,replacement_path=save_path)
    pipe.register_attn(replacement_location,attn)
    seed_everything(seed)
    out = pipe(
        device_map=device_map,
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        latents=start_latents,
        num_inference_steps=steps,
        first_order=first_order,
        path=save_path
    )
    reg_save_mid_step(pipe,False)

    x0 = out["end_latents"]
    xT = out["latents"]
    img = out['images']

    torch.save(xT,os.path.join(save_path,f'latents_{steps}.pt'))
    
    return x0, xT, img
