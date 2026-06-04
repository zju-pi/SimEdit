from diffusers import FluxPipeline
from IPython.display import display
import torchvision.transforms as T
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
import torch
from typing import Union, List, Optional, Dict

from diffusers.models.attention_processor import Attention
import copy
import numpy as np

def show_image(tensor: torch.Tensor):
    img = T.ToPILImage()(tensor.float())
    display(img)
    
class ParallelFluxPipeline(FluxPipeline):
    """
    Unified Flux Pipeline supporting multi-device parallel execution.
    Combines functionality from FlexibleFluxPipeline and ParallelFluxPipeline.
    """
    
    def get_latents_input(self,latents,t):
        return latents

    def register(self,location,callback):
        location = location.split('.')
        n = len(location)
        num_reg = 0
        arr_stack = [] # layer and block record
        where = location[0]
        def get_model(model,idx):
            nonlocal num_reg
            if idx == n:
                callback(model,copy.deepcopy(arr_stack),where)
                num_reg += 1
                return
            if location[idx][0] == '[':
                for i in eval(location[idx]):
                    arr_stack.append(i)
                    get_model(model[i],idx + 1)
                    del arr_stack[-1]
            else:
                get_model(getattr(model,location[idx]),idx + 1)
        
        get_model(self.transformer,0)
        return num_reg
    
    def register_time(self,location,t,ti):
        def reg_time(model,arr_stack,where):
            setattr(model,'t',t)
            setattr(model,'ti',ti)
        self.register(location,reg_time)
    
    def register_attn(self,locaiton,editor):#: AttentionBase
        def set_forward(attn: Attention,arr_stack,where):
            """
            Attention forward function, refer from diffusers v0.31.0 FluxAttnProcessor2_0
            """
            def forward(
                hidden_states: torch.FloatTensor, # [batch, seq_len, heads=24*head_dim=128] seq_len= H * W  in double | H * W + prompt_embd_seq_len in single
                encoder_hidden_states: torch.FloatTensor = None, # [batch, prompt_embd_seq_len=512, heads=24*head_dim=128]
                attention_mask: Optional[torch.FloatTensor] = None,
                image_rotary_emb: Optional[torch.Tensor] = None,
                ) -> torch.FloatTensor:

                # edit attn input
                if hasattr(editor,'input'):
                    hidden_states, encoder_hidden_states = editor.input(hidden_states, encoder_hidden_states)

                batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

                # `sample` projections.
                query = attn.to_q(hidden_states) # [batch, seq_len, heads=24*head_dim=128]
                key = attn.to_k(hidden_states) # [batch, seq_len, heads=24*head_dim=128]
                value = attn.to_v(hidden_states) # [batch, seq_len, heads=24*head_dim=128]

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [batch, heads, seq_len ,head_dim]
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [batch, heads, seq_len ,head_dim]
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [batch, heads, seq_len ,head_dim]

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                is_single = True
                # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
                if encoder_hidden_states is not None:
                    is_single = False
                    # `context` projections.
                    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) # [batch, prompt_embd_seq_len, heads=24*head_dim=128]
                    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states) # [batch, prompt_embd_seq_len, heads=24*head_dim=128]
                    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) # [batch, prompt_embd_seq_len, heads=24*head_dim=128]

                    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2) # [batch, heads, prompt_embd_seq_len ,head_dim]
                    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2) # [batch, heads, prompt_embd_seq_len ,head_dim]
                    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2) # [batch, heads, prompt_embd_seq_len ,head_dim]

                    if attn.norm_added_q is not None:
                        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                    if attn.norm_added_k is not None:
                        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

                    # attention
                    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2) # [batch, heads, prompt_embd_seq_len + seq_len, head_dim]
                    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2) # [batch, heads, prompt_embd_seq_len + seq_len, head_dim]
                    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2) # [batch, heads, prompt_embd_seq_len + seq_len, head_dim]

                if image_rotary_emb is not None:
                    from diffusers.models.embeddings import apply_rotary_emb

                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)


                # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                
                sim = torch.einsum('b h i d , b h j d -> b h i j',query,key) * attn.scale
                attn_map = sim.softmax(dim=-1)
                # hidden_states = torch.einsum('b h i j,b h j d -> b h i d',attn_map, value)

                hidden_states = editor(
                    hidden_states, is_single, encoder_hidden_states, attention_mask,
                    query, key, value,
                    sim, attn_map,
                    attn.heads, attn.scale,
                    attn, arr_stack, where
                )
                
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

                if encoder_hidden_states is not None:
                    encoder_hidden_states, hidden_states = (
                        hidden_states[:, : encoder_hidden_states.shape[1]],
                        hidden_states[:, encoder_hidden_states.shape[1] :],
                    )

                    # linear proj
                    hidden_states = attn.to_out[0](hidden_states)
                    # dropout
                    hidden_states = attn.to_out[1](hidden_states)
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states
                else:
                    return hidden_states
                
            
            attn.forward = forward
        self.register(locaiton,set_forward)
        
    @torch.no_grad()
    def prepare_timesteps(self,num_inference_steps,image_seq_len,device,timesteps=None):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @torch.no_grad()
    def encode_latents(self,init_image: torch.Tensor):
        init_image = init_image * 2 - 1
        init_image = init_image.unsqueeze(0).to(self.vae.device)
        posterior = self.vae.encode(init_image.to(torch.bfloat16), return_dict=False)[0]
        latents = (posterior.mean - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode_latents(self,latents,height=None,width=None): 
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pt')
        return image

    @torch.no_grad()
    def __call__(
        self,
        device_map: Dict[str,str],
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        align_batch=False,
        first_order=True,
        start_steps=0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        **kargs
    ):  
        for i in kargs:
            setattr(self,i,kargs[i])
        self.prompt = prompt

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        print(f'vae_scale_factor={self.vae_scale_factor}')

        self._guidance_scale = guidance_scale
        self._interrupt = False

        time_location = [
            f"transformer_blocks.[{','.join([str(i) for i in range(19)])}].attn",
            f"single_transformer_blocks.[{','.join([str(i) for i in range(38)])}].attn"
        ]
        self.time_location = time_location

        # 1. Prompt -> Embedding
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.device_map = device_map

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device_map['text_encoder'],
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 2. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        if align_batch:
            shape = (batch_size * num_images_per_prompt, num_channels_latents, 2 * (int(height) // (self.vae_scale_factor * 2)), 2 * (int(width) // (self.vae_scale_factor * 2)))
            latents = randn_tensor((1,*shape[1:]), generator=generator, device=device_map['vae'], dtype=prompt_embeds.dtype)
            latents = latents.expand(shape)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, 2 * (int(height) // (self.vae_scale_factor * 2)), 2 * (int(width) // (self.vae_scale_factor * 2)))

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device_map['vae'],
            generator,
            latents,
        )
        # show_image(self.decode_latents(latents)[0])

        # 3. Prepare timesteps
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps,image_seq_len,device_map['vae'],timesteps)
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 4. handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device_map['transformer'], dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None


        latent_image_ids = latent_image_ids.to(device_map['transformer'])
        pooled_prompt_embeds = pooled_prompt_embeds.to(device_map['transformer'])
        text_ids = text_ids.to(device_map['transformer'])

        self.guidance = guidance
        self.latent_image_ids = latent_image_ids
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids

        output = {
            "latents": latents
        }

        # 5. Denoising loop
        timesteps = torch.cat([timesteps,torch.tensor([0.],device=timesteps.device)])
        self.timesteps = timesteps
        with self.progress_bar(total=num_inference_steps-start_steps) as progress_bar:
            for i, t in enumerate(timesteps[start_steps:-1]):
                if self.interrupt:
                    continue
                # print(i+start_steps,i+start_steps+1, timesteps[i+start_steps], timesteps[i+start_steps+1])
                # step
                latents = self.step(latents,i+start_steps,t,timesteps[i+start_steps+1],first_order)
                
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()
                progress_bar.update()
                

        
        output["end_latents"] = latents
        output["images"] = self.decode_latents(latents,height,width)

        # Offload all models
        self.maybe_free_model_hooks()

        return output
    
    @torch.no_grad()
    def get_prompt_embeds(self, prompt, num_images_per_prompt: Optional[int] = 1,max_sequence_length: int = 512):
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt, 
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=self.device_map['text_encoder'],
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    @torch.no_grad()
    def invert(
        self,
        device_map: Dict[str,str],
        image: torch.FloatTensor,
        prompt: str,
        prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 1,
        first_order=True,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        inversion_method: str = "fixed_point",  # "euler" or "fixed_point"
        **kargs
    ):
        for i in kargs:
            setattr(self,i,kargs[i])
        self.prompt = prompt

        height = image.shape[1]
        width = image.shape[2]

        print(f'vae_scale_factor={self.vae_scale_factor}')

        self._guidance_scale = guidance_scale
        self._interrupt = False

        time_location = [
            f"transformer_blocks.[{','.join([str(i) for i in range(19)])}].attn",
            f"single_transformer_blocks.[{','.join([str(i) for i in range(38)])}].attn"
        ]
        self.time_location = time_location

        # 1. Prompt -> Embedding
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.device_map = device_map

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device_map['text_encoder'],
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        # 2. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        
        latents = self.encode_latents(image)
        # print(latents.shape)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, latents.shape[2], latents.shape[3])
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device_map['vae'],
            generator,
            latents,
        )

        # 3. Prepare timesteps
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps,image_seq_len,device_map['vae'],timesteps)
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 4. handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device_map['transformer'], dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        latent_image_ids = latent_image_ids.to(device_map['transformer'])
        pooled_prompt_embeds = pooled_prompt_embeds.to(device_map['transformer'])
        text_ids = text_ids.to(device_map['transformer'])

        self.guidance = guidance
        self.latent_image_ids = latent_image_ids
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids

        output = {
            "start_latents": latents
        }

        # 5. Denoising loop
        timesteps = torch.cat([timesteps,torch.tensor([0.],device=timesteps.device)])
        # reverse when invertion
        timesteps = reversed(timesteps)
        self.timesteps = timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue
                
                if inversion_method == "euler":
                    # Traditional Euler method (from diffuser.py)
                    latents = self.step(latents,i,t,timesteps[i+1],first_order,inverse=True)
                elif inversion_method == "fixed_point":
                    # Fixed-point inversion method (from diffuser2.py)
                    prev_latents = latents
                    latents = self.step(latents,i,t,timesteps[i+1],first_order,inverse=True,t_cur_=t)
                    latents = self.step(latents,i,t,timesteps[i+1],first_order,inverse=True,prev_latents=prev_latents,t_cur_=timesteps[i+1])
                else:
                    raise ValueError(f"Unknown inversion_method: {inversion_method}. Use 'euler' or 'fixed_point'.")
                
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()
                progress_bar.update()

        output["latents"] = latents
        output["images"] = self.decode_latents(latents,height,width)

        # Offload all models
        self.maybe_free_model_hooks()

        return output
    
    @torch.no_grad()
    def step(self,latents,i,t_cur,t_prev,first_order=True,inverse=False,prev_latents=None,t_cur_=None):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if t_cur_ is None:
            t_cur_ = t_cur
        timestep = t_cur_.expand(latents.shape[0]).to(latents.dtype)

        latents = self.get_latents_input(latents,t_cur)
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

        return latents