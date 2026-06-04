import os
import glob
import torch
import joblib
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================================
# Layer 1: Basic Components (Storage Components)
# ============================================================================

class AttentionStorage:
    """
    Minimalist Attention Map storage class
    Responsibility: Only handles checking and saving, no computation
    """
    
    def __init__(self,
                 save_dir: str = 'attn_map',
                 save_type: str = 'avg_t'):
        """
        Args:
            save_dir: Save directory
            save_type: Save type, one of three options:
                       'every': Save each attention map
                       'avg_t': Average by timestep and save
                       'avg_all': Average across all timesteps and save
        """
        # Validate save_type parameter
        valid_save_types = ['every', 'avg_t', 'avg_all']
        if save_type not in valid_save_types:
            raise ValueError(f"save_type must be one of {valid_save_types}, got '{save_type}'")
        
        self.save_dir = save_dir
        self.save_type = save_type
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.avg_attn_map = {}
        self.cnt = {}
        self.last_t = {}
        self.last_t['single_transformer_blocks'] = -1
        self.last_t['transformer_blocks'] = -1
    
    def save(self, attn: torch.Tensor, where: str, timestep: Optional[int] = None, layer: Optional[int] = None, fo: Optional[int] = None):
        """
        Save attention map
        Minimal logic: just save, nothing more
        """
        if not self.save_dir:
            return
        if timestep == None:
            filename = f'avg_all_attn_{where}.pt'
        elif layer == None:
            filename = f'avg_t_attn_{where}_T{timestep}.pt'
        else:
            filename = f'attn_{where}_T{timestep}_L{layer}_fo{fo}.pt'
        filepath = os.path.join(self.save_dir, filename)
        torch.save(attn.cpu(), filepath)

    def reg(self, attn: torch.Tensor, where: str, timestep: int, layer: int, fo: int):
        """
        Register attention map
        Save and process according to the configured method
        """
        if self.save_type == 'every':
            self.save(attn, where, timestep, layer, fo)
        elif self.save_type == 'avg_t':
            if timestep != self.last_t[where]:
                if self.last_t[where] != -1:
                    avg = self.avg_attn_map[where] / self.cnt[where]
                    self.save(avg, where, self.last_t[where])
                self.avg_attn_map[where] = attn.clone()
                self.cnt[where] = 1
            else:
                self.avg_attn_map[where] += attn
                self.cnt[where] += 1
            self.last_t[where] = timestep
        elif self.save_type == 'avg_all':
            if where not in self.avg_attn_map:
                self.avg_attn_map[where] = attn.clone()
                self.cnt[where] = 1
            else:
                self.avg_attn_map[where] += attn
                self.cnt[where] += 1
        
    def flush(self):
        if self.save_type == 'avg_t':
            for where in self.avg_attn_map:
                if self.cnt[where] > 0:  # Has data but not saved
                    avg = self.avg_attn_map[where] / self.cnt[where]
                    self.save(avg, where, self.last_t[where])
        elif self.save_type == 'avg_all':
            for where in self.avg_attn_map:
                if self.cnt[where] > 0:  # Has data but not saved
                    avg = self.avg_attn_map[where] / self.cnt[where]
                    self.save(avg, where)


class AttentionCollector:
    """
    Minimalist Attention Map storage class
    Responsibility: Only handles checking and saving, no computation
    """
    
    def __init__(self):
        """
        Args:
            save_dir: Save directory
            save_type: Save type, one of three options:
                       'every': Save each attention map
                       'avg_t': Average by timestep and save
                       'avg_all': Average across all timesteps and save
        """
        
        self.avg_attn_map = {}
        self.cnt = {}
        self.last_t = {}
        self.last_t['single_transformer_blocks'] = -1
        self.last_t['transformer_blocks'] = -1
    

    def reg(self, attn: torch.Tensor, where: str, timestep: int, layer: int, fo: int):
        """
        Register attention map
        Save and process according to the configured method
        """
        if timestep != self.last_t[where]:
            self.avg_attn_map[where] = attn.clone()
            self.cnt[where] = 1
        else:
            self.avg_attn_map[where] += attn
            self.cnt[where] += 1
        self.last_t[where] = timestep
    
    def get_current(self, where: str) -> Optional[torch.Tensor]:
        if self.last_t[where] == -1:
             raise ValueError(f"No attention map")
        else:
            avg = self.avg_attn_map[where] / self.cnt[where]
            return avg
    
    def clear(self):
        self.avg_attn_map = {}
        self.cnt = {}
        self.last_t = {}
        self.last_t['single_transformer_blocks'] = -1
        self.last_t['transformer_blocks'] = -1



class QKVStorage:
    """
    Storage and loading of Q, K, V
    Responsibility: Only handles QKV persistence
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 use_disk: bool = True):
        """
        Args:
            storage_path: Storage path, None means memory storage
            use_disk: Whether to use disk storage (False means memory storage)
        """
        self.storage_path = storage_path
        self.use_disk = use_disk
        self.memory_cache = {}  # Memory cache
        
        if use_disk and storage_path:
            os.makedirs(storage_path, exist_ok=True)
    
    def _make_key(self, where: str, layer: int, fo: int, timestep: int) -> str:
        """Generate storage key"""
        return f'{where}_{layer}_{fo}_{timestep}'
    
    def save_qkv(self, q, k, v, where: str, layer: int, fo: int, timestep: int):
        """Save Q, K, V"""
        key = self._make_key(where, layer, fo, timestep)
        
        if self.use_disk and self.storage_path:
            filepath = os.path.join(self.storage_path, f'{key}.pt')
            joblib.dump((q.cpu(), k.cpu(), v.cpu()), filepath)
        else:
            # Memory storage
            self.memory_cache[key] = (q.cpu(), k.cpu(), v.cpu())
    
    def load_qkv(self, where: str, layer: int, fo: int, timestep: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load Q, K, V"""
        key = self._make_key(where, layer, fo, timestep)
        
        if self.use_disk and self.storage_path:
            filepath = os.path.join(self.storage_path, f'{key}.pt')
            return joblib.load(filepath)
        else:
            # Load from memory
            raise ValueError("No matching files found")
            return self.memory_cache[key]
    
    def exists(self, where: str, layer: int, fo: int, timestep: int) -> bool:
        """Check if exists"""
        key = self._make_key(where, layer, fo, timestep)
        
        if self.use_disk and self.storage_path:
            filepath = os.path.join(self.storage_path, f'{key}.pt')
            return os.path.exists(filepath)
        else:
            return key in self.memory_cache


# ============================================================================
# Layer 2: Strategy Layer (Strategies)
# ============================================================================

class QKVReplacementStrategy:
    """
    Q, K, V replacement strategy
    Simple linear mixing
    """
    
    def __init__(self, mix_ratios: Tuple[float, float, float] = (0, 0, 1)):
        """
        Args:
            mix_ratios: (a, b, c) corresponding to mixing ratios for Q, K, V respectively
                       result = ratio * source + (1 - ratio) * target
        """
        self.a, self.b, self.c = mix_ratios
    
    def apply(self, 
              q_tgt, k_tgt, v_tgt,
              q_src, k_src, v_src) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mix source and target Q, K, V
        """
        q = self.a * q_src.to(q_tgt.device) + (1 - self.a) * q_tgt
        k = self.b * k_src.to(k_tgt.device) + (1 - self.b) * k_tgt
        v = self.c * v_src.to(v_tgt.device) + (1 - self.c) * v_tgt
        return q, k, v


class SpecialTokenReplaceStrategy:
    """
    Special token replacement logic
    Handles separate replacement of text tokens and image tokens
    """
    
    def __init__(self,
                 spec_replace_id: Tuple[List[int], List[int]],
                 text_token_count: int = 512,
                 mix_ratios: Tuple[float, float, float] = (0, 0, 1)):
        """
        Args:
            spec_replace_id: (target_ids, source_ids)
                target_ids: Token positions in target prompt that need to be replaced
                source_ids: Corresponding token positions in source prompt
            text_token_count: Number of text tokens (used to separate text and image tokens)
            mix_ratios: Mixing ratios
        """
        self.target_ids = spec_replace_id[0]
        self.source_ids = spec_replace_id[1]
        self.text_token_count = text_token_count
        self.a, self.b, self.c = mix_ratios
    
    def apply(self,
              q_tgt, k_tgt, v_tgt,
              q_src, k_src, v_src) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separate replacement:
        1. text tokens (first 512): Only replace positions specified by spec_replace_id
        2. image tokens (remaining): Replace all
        """
        # Separate text and image tokens
        Qp = q_tgt[:, :, :self.text_token_count, :].clone()
        Kp = k_tgt[:, :, :self.text_token_count, :].clone()
        Vp = v_tgt[:, :, :self.text_token_count, :].clone()
        
        # Replace all image tokens
        Ql = self.a * q_src[:, :, self.text_token_count:, :].to(q_tgt.device) + \
             (1 - self.a) * q_tgt[:, :, self.text_token_count:, :]
        Kl = self.b * k_src[:, :, self.text_token_count:, :].to(k_tgt.device) + \
             (1 - self.b) * k_tgt[:, :, self.text_token_count:, :]
        Vl = self.c * v_src[:, :, self.text_token_count:, :].to(v_tgt.device) + \
             (1 - self.c) * v_tgt[:, :, self.text_token_count:, :]
        
        # Replace specific positions in text tokens
        q_src_text = q_src[:, :, :self.text_token_count, :]
        k_src_text = k_src[:, :, :self.text_token_count, :]
        v_src_text = v_src[:, :, :self.text_token_count, :]
        
        Qp[:, :, self.target_ids, :] = self.a * q_src_text[:, :, self.source_ids, :].to(q_tgt.device) + \
                                        (1 - self.a) * Qp[:, :, self.target_ids, :]
        Kp[:, :, self.target_ids, :] = self.b * k_src_text[:, :, self.source_ids, :].to(k_tgt.device) + \
                                        (1 - self.b) * Kp[:, :, self.target_ids, :]
        Vp[:, :, self.target_ids, :] = self.c * v_src_text[:, :, self.source_ids, :].to(v_tgt.device) + \
                                        (1 - self.c) * Vp[:, :, self.target_ids, :]
        
        # Concatenate
        q = torch.cat([Qp, Ql], dim=2)
        k = torch.cat([Kp, Kl], dim=2)
        v = torch.cat([Vp, Vl], dim=2)
        
        return q, k, v


class AttentionModulationStrategy:
    """
    Attention map modulation strategy
    Used to enhance or weaken attention for specific tokens
    """
    
    def __init__(self,
                 emphasize_scale: float = 1.0,
                 max_token_id: int = 512,
                 exclude_token_ids: Optional[List[int]] = None):
        """
        Args:
            emphasize_scale: Scaling factor
            max_token_id: Token range to modulate [0, max_token_id)
            exclude_token_ids: Token ids to exclude (no modulation applied)
        """
        self.emphasize_scale = emphasize_scale
        self.max_token_id = max_token_id
        self.exclude_token_ids = set(exclude_token_ids) if exclude_token_ids else set()
    
    def apply(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Modulate attention map
        attn: [batch, heads, seq_len, seq_len]
        """
        if self.emphasize_scale == 1.0:
            return attn
        
        attn = attn.clone()
        for token_id in range(0, self.max_token_id):
            if token_id not in self.exclude_token_ids:
                attn[:, :, :, token_id] = attn[:, :, :, token_id] * self.emphasize_scale
        
        return attn


# ============================================================================
# Layer 3: Main Processors (Store/Replacement/Optimization)
# ============================================================================


class AttentionBase:
    """
    Original attention mechanism
    """ 
    
    def __call__(self,
                 hidden_states, is_single, encoder_hidden_states, attention_mask,
                 q, k, v,
                 sim, attn,
                 heads, scale,
                 model, pos, where):

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out


class AttentionReplacement:
    """
    Full-featured Attention class
    Includes QKV replacement, Emphasize, Special Replace
    """
    
    def __init__(self,
                 inject_timestep,
                 replacement_path: str = None,
                 max_token_id: int = 512,
                 emphasize_scale: float = 1.0,
                 mix_ratios: Tuple[float, float, float] = (0, 0, 1),
                 spec_replace_id: Optional[Tuple[List[int], List[int]]] = None):
       
        self.inject_timestep = inject_timestep
        self.replacement_path = replacement_path  
        self.max_token_id = max_token_id
        self.emphasize_scale = emphasize_scale   

        self.qkv_storage = QKVStorage(
            storage_path=replacement_path,
            use_disk=replacement_path is not None
        )
        # Create modulation strategy
        self.modulation_strategy = AttentionModulationStrategy(
            emphasize_scale=emphasize_scale,
            max_token_id=max_token_id,
            exclude_token_ids=set(spec_replace_id[0]) if spec_replace_id else None
        )
        
        # Attributes for configuration (maintain compatibility)
        self.a, self.b, self.c = mix_ratios
        self.spec_replace_id = spec_replace_id

        self.replacement_strategy = QKVReplacementStrategy(mix_ratios=(self.a, self.b, self.c))

    def load_replacement_qkv(self, where, pos, fo, t):
        layer = pos[0] if isinstance(pos, tuple) else pos
        return self.qkv_storage.load_qkv(where, layer, fo, t)
    
    
    def __call__(self,
                 hidden_states, is_single, encoder_hidden_states, attention_mask,
                 q, k, v,
                 sim, attn,
                 heads, scale,
                 model, pos, where):
        """Complete processing pipeline"""
        t = model.t if hasattr(model, 't') else 0
        fo = model.fo if hasattr(model, 'fo') else 0
        
        if t in self.inject_timestep:
            
            q_src, k_src, v_src = self.load_replacement_qkv(where, pos, fo, t)
            
            # # Update strategy mixing ratios
            # self.replacement_strategy.a = self.a
            # self.replacement_strategy.b = self.b
            # self.replacement_strategy.c = self.c
            
            if not hasattr(self, 'spec_replace_id') or self.spec_replace_id is None:
                # Simple replacement
                q, k, v = self.replacement_strategy.apply(q, k, v, q_src, k_src, v_src)
            else:
                # Special replacement
                special_strategy = SpecialTokenReplaceStrategy(
                    spec_replace_id=self.spec_replace_id,
                    text_token_count=512,
                    mix_ratios=(self.a, self.b, self.c)
                )
                q, k, v = special_strategy.apply(q, k, v, q_src, k_src, v_src)
            
            # Recompute attention
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale
            attn = sim.softmax(dim=-1)
        
        # Attention modulation
        # Note: Only apply emphasize when spec_replace_id exists (maintain compatibility with attention2)
        if hasattr(self, 'spec_replace_id') and self.spec_replace_id is not None:
            attn = self.modulation_strategy.apply(attn)
        
        # Compute output
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out

class AttentionOptimization:
    """
    Attention class specifically for opt function
    Only performs attention modulation and collection, no QKV replacement
    """
    
    def __init__(self,
                 max_token_id: int = 512,
                 emphasize_scale: float = 1.0,
                 spec_replace_id: Optional[Tuple[List[int], List[int]]] = None):
        """
        Args:
            max_token_id: Token range to modulate
            emphasize_scale: Scaling factor
            spec_replace_id: Token positions for special replacement
            keep_grad: Whether to keep gradients (for loss calculation)
        """
        self.max_token_id = max_token_id
        self.emphasize_scale = emphasize_scale
        self.spec_replace_id = spec_replace_id
        
        # Create modulation strategy
        self.modulation_strategy = AttentionModulationStrategy(
            emphasize_scale=emphasize_scale,
            max_token_id=max_token_id,
            exclude_token_ids=set(spec_replace_id[0]) if spec_replace_id else None
        )
        
        # Create collector
        self.attn_collector = AttentionCollector()
    
    def get(self, where: str) -> Optional[torch.Tensor]:
        return self.attn_collector.get_current(where)
    
    def __call__(self,
                 hidden_states, is_single, encoder_hidden_states, attention_mask,
                 q, k, v,
                 sim, attn,
                 heads, scale,
                 model, pos, where):
        """Processing pipeline: modulation + collection"""
        t = model.t if hasattr(model, 't') else 0
        layer = pos[0] if isinstance(pos, tuple) else pos
        fo = model.fo if hasattr(model, 'fo') else 0
        # Apply attention modulation
        if self.spec_replace_id is not None:
            attn = self.modulation_strategy.apply(attn)
        # Register attention map
        self.attn_collector.reg(attn, where, t, layer, fo)
        
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


class StoreAttention:

    def store_replacement_qkv(self, q, k, v, where, pos, fo, t):
        layer = pos[0] if isinstance(pos, tuple) else pos
        self.qkv_storage.save_qkv(q, k, v, where, layer, fo, t)
    
    def __init__(self,
                 inject_timestep,
                 store_attn: bool = True,
                 store_qkv: bool = True,
                 attn_map_path: str = 'attn_map',
                 save_type: str = 'avg_t',
                 replacement_path: str = None):
        self.inject_timestep = inject_timestep
        self.replacement_path = replacement_path  
        self.store_attn = store_attn
        self.store_qkv = store_qkv

        if self.store_attn == True:
            self.attn_storage = AttentionStorage(
                save_dir=attn_map_path,
                save_type=save_type
            )

        self.qkv_storage = QKVStorage(
            storage_path=replacement_path,
            use_disk=replacement_path is not None
        )
    def flush(self):
        self.attn_storage.flush()
    def __call__(self,
                 hidden_states, is_single, encoder_hidden_states, attention_mask,
                 q, k, v,
                 sim, attn,
                 heads, scale,
                 model, pos, where):
    
        t = model.t if hasattr(model, 't') else 0
        fo = model.fo if hasattr(model, 'fo') else 0

        if t in self.inject_timestep and self.store_qkv == True:
            self.store_replacement_qkv(q, k, v, where, pos, fo, t)
        
        layer = pos[0] if isinstance(pos, tuple) else pos
        if self.store_attn == True:
            self.attn_storage.reg(attn, where, t, layer, fo)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out