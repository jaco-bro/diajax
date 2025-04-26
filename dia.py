import functools
import os
from typing import List, Dict, Union, Optional, Tuple, Callable, Any
from pydantic import BaseModel, Field
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
import dac # try DAC-JAX
import torch

class DataConfig(BaseModel):
    text_length: int = Field(gt=0)
    audio_length: int = Field(gt=0)
    channels: int = Field(default=9, gt=0)
    text_pad_value: int = Field(default=0)
    audio_eos_value: int = Field(default=1024)
    audio_pad_value: int = Field(default=1025)
    audio_bos_value: int = Field(default=1026)
    delay_pattern: list[int] = Field(default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15])

class EncoderConfig(BaseModel):
    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    n_head: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)

class DecoderConfig(BaseModel):
    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    gqa_query_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    gqa_head_dim: int = Field(gt=0)
    cross_query_heads: int = Field(gt=0)
    cross_head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)

class ModelConfig(BaseModel):
    encoder: EncoderConfig
    decoder: DecoderConfig
    src_vocab_size: int = Field(default=128, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    weight_dtype: str = Field(default="float32")
    rope_min_timescale: int = Field(default=1)
    rope_max_timescale: int = Field(default=10_000)

class TrainingConfig(BaseModel):
    dtype: str = Field(default="float32")
    logits_dot_in_fp32: bool = Field(default=False)

class DiaConfig(BaseModel):
    version: str = Field(default="1.0")
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def load(cls, path: str):
        try:
            with open(path, "r") as f:
                import json
                content = f.read()
                return cls.model_validate(json.loads(content))
        except FileNotFoundError:
            return None
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        config_dict = self.model_dump()
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

def build_revert_indices(B, T, C, delay_pattern):
    delay_arr = jnp.array(delay_pattern, dtype=jnp.int32)
    t_idx_BT1 = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32).reshape(1, -1, 1), [B, T, 1])
    t_idx_BxTxC = jnp.minimum(t_idx_BT1 + delay_arr.reshape(1, 1, C), jnp.array(T - 1, dtype=jnp.int32))
    b_idx_BxTxC = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32).reshape(B, 1, 1), [B, T, C])
    c_idx_BxTxC = jnp.broadcast_to(jnp.arange(C, dtype=jnp.int32).reshape(1, 1, C), [B, T, C])
    indices_BTCx3 = jnp.stack([b_idx_BxTxC.reshape(-1), t_idx_BxTxC.reshape(-1), c_idx_BxTxC.reshape(-1)], axis=1)
    return t_idx_BxTxC, indices_BTCx3

def revert_audio_delay(audio_BxTxC, pad_value, precomp, T):
    t_idx_BxTxC, indices_BTCx3 = precomp
    b_indices = indices_BTCx3[:, 0]
    t_indices = indices_BTCx3[:, 1]
    c_indices = indices_BTCx3[:, 2]
    gathered_flat = audio_BxTxC[b_indices, t_indices, c_indices]
    gathered_BxTxC = gathered_flat.reshape(audio_BxTxC.shape)
    mask_pad = t_idx_BxTxC >= T
    result_BxTxC = jnp.where(mask_pad, jnp.full_like(gathered_BxTxC, pad_value), gathered_BxTxC)
    return result_BxTxC

def codebook_to_audio(generated_codes, delay_pattern, B=1, T=2600, C=9):
    generated_codes = generated_codes[:, 1:]
    if generated_codes.shape[1] > T:
        generated_codes = generated_codes[:, :T]
    seq_length = generated_codes.shape[1]
    t_idx_BxTxC, indices_BTCx3 = build_revert_indices(B=B, T=seq_length, C=C, delay_pattern=delay_pattern)
    audio_BxTxC = generated_codes.transpose(1, 0)[None, ...]
    reverted_codebook = revert_audio_delay(audio_BxTxC=audio_BxTxC, pad_value=0, precomp=(t_idx_BxTxC, indices_BTCx3), T=seq_length)
    reverted_codebook = reverted_codebook[:, :-30, :]
    codebook = reverted_codebook.transpose(0, 2, 1)
    min_valid_index = 0
    max_valid_index = 1023
    invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
    codebook = jnp.where(invalid_mask, jnp.zeros_like(codebook), codebook)
    try:
        with torch.no_grad(), torch.inference_mode():
            model = dac.DAC.load(dac.utils.download())
            device = model.device
            codebook_torch = torch.from_numpy(np.array(codebook)).long().to(device)
            audio_values = model.quantizer.from_codes(codebook_torch)
            audio_values = model.decode(audio_values[0])
            return audio_values.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error in decode: {e}")
        return np.zeros((1, 44100))

@nnx.jit
def apply_rope(k, cos, sin):
    k1, k2 = jnp.split(k, 2, axis=-1)
    return jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

@nnx.jit
def create_attention_mask(q_padding_mask, k_padding_mask):
    B1, Tq = q_padding_mask.shape
    B2, Tk = k_padding_mask.shape
    q_mask_BxTqx1 = q_padding_mask[:, :, None]
    k_mask_Bx1xTk = k_padding_mask[:, None, :]
    non_pad_attends_non_pad = q_mask_BxTqx1 & k_mask_Bx1xTk
    pad_attends_pad = (~q_mask_BxTqx1) & (~k_mask_Bx1xTk)
    mask = non_pad_attends_non_pad | pad_attends_pad
    mask = jnp.where(mask, 0.0, -1e9)
    return mask[:, None, :, :]

class Roper(nnx.Module):
    def __init__(self, head_dim, min_timescale=1, max_timescale=10000):
        fraction = (2.0 * jnp.arange(0, head_dim//2, dtype=jnp.float32)) / head_dim
        self.timescale = nnx.Variable(min_timescale * (max_timescale / min_timescale) ** fraction)
        
    @nnx.jit
    def __call__(self, positions):
        positions = positions[:, :, None, None]
        sinusoid_inp = positions / self.timescale
        cos = jnp.cos(sinusoid_inp) 
        sin = jnp.sin(sinusoid_inp)
        return cos, sin

class MlpBlock(nnx.Module):
    def __init__(self, config, embed_dim, intermediate_dim, use_pre_norm=False, *, rngs: nnx.Rngs):
        self.wi_fused = nnx.LinearGeneral(in_features=embed_dim, out_features=(2, intermediate_dim), axis=-1, use_bias=False, rngs=rngs)
        self.wo = nnx.LinearGeneral(in_features=intermediate_dim, out_features=embed_dim, axis=-1, use_bias=False, rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        fused_x = self.wi_fused(x)
        return self.wo(nnx.silu(fused_x[..., 0, :]) * fused_x[..., 1, :])

class Attention(nnx.Module):
    def __init__(self, config, q_embed_dim, kv_embed_dim, num_query_heads, num_kv_heads, head_dim, dropout_rate, is_cross_attn=False, out_embed_dim=None, *, rngs: nnx.Rngs):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        self.dropout_rate = dropout_rate
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        self.num_gqa_groups = num_query_heads // num_kv_heads
        self.q_proj = nnx.LinearGeneral(in_features=q_embed_dim, out_features=(num_query_heads, head_dim), axis=-1, use_bias=False, rngs=rngs)
        self.k_proj = nnx.LinearGeneral(in_features=kv_embed_dim, out_features=(num_kv_heads, head_dim), axis=-1, use_bias=False, rngs=rngs)
        self.v_proj = nnx.LinearGeneral(in_features=kv_embed_dim, out_features=(num_kv_heads, head_dim), axis=-1, use_bias=False, rngs=rngs)
        self.o_proj = nnx.LinearGeneral(in_features=(num_query_heads, head_dim), out_features=self.output_dim, axis=(-2, -1), use_bias=False, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x, rope_cos, rope_sin, attn_mask=None, kv_cache=None):
        q = self.q_proj(x)
        q = apply_rope(q, rope_cos, rope_sin)
        q = jnp.transpose(q, (0, 2, 1, 3))
        if self.is_cross_attn:
            k, v = kv_cache
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
            k = apply_rope(k, rope_cos, rope_sin)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))
            if self.num_gqa_groups > 1:
                k = jnp.repeat(k, self.num_gqa_groups, axis=1)
                v = jnp.repeat(v, self.num_gqa_groups, axis=1)
            if kv_cache is not None:
                k = jnp.concatenate([kv_cache[0], k], axis=2)
                v = jnp.concatenate([kv_cache[1], v], axis=2)
        w = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        if attn_mask is not None:
            w = w + attn_mask
        w = nnx.softmax(w, axis=-1)
        w = jnp.matmul(w, v)
        w = jnp.transpose(w, (0, 2, 1, 3))
        output = self.o_proj(w)
        return output, (k, v)

class EncoderLayer(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd
        self.pre_sa_norm = nnx.RMSNorm(num_features=embed_dim, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.self_attention = Attention(config=config, q_embed_dim=embed_dim, kv_embed_dim=embed_dim, num_query_heads=enc_config.n_head, num_kv_heads=enc_config.n_head, head_dim=enc_config.head_dim, dropout_rate=model_config.dropout, is_cross_attn=False, out_embed_dim=embed_dim, rngs=rngs)
        self.post_sa_norm = nnx.RMSNorm(num_features=embed_dim, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.mlp = MlpBlock(config=config, embed_dim=embed_dim, intermediate_dim=enc_config.n_hidden, use_pre_norm=enc_config.use_pre_norm, rngs=rngs)
        self.dropout_rate = model_config.dropout
   
    @nnx.jit
    def __call__(self, x, attn_mask=None, rope_cos=None, rope_sin=None):
        residual = x
        sa_out, _ = self.self_attention(x=self.pre_sa_norm(x), rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=attn_mask)        
        x = residual + sa_out
        residual = x
        return residual + self.mlp(self.post_sa_norm(x))
        return x

class Encoder(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        model_config = config.model
        enc_config = config.model.encoder
        self.embedding = nnx.Embed(
            num_embeddings=model_config.src_vocab_size,
            features=enc_config.n_embd,
            rngs=rngs,
        )
        self.layers = [
            EncoderLayer(config=config, rngs=rngs) 
            for _ in range(enc_config.n_layer)
        ]
        self.norm = nnx.RMSNorm(
            num_features=enc_config.n_embd,
            epsilon=model_config.normalization_layer_epsilon,
            rngs=rngs,
        )
        self.dropout_rate = model_config.dropout

    @nnx.jit
    def __call__(self, x_ids, attn_mask=None, rope_cos=None, rope_sin=None):
        x = self.embedding(x_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        return x

class DecoderLayer(nnx.Module):
    def __init__(self, config: DiaConfig, *, rngs: nnx.Rngs):
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd
        self.pre_sa_norm = nnx.RMSNorm(num_features=dec_embed_dim, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.pre_ca_norm = nnx.RMSNorm(num_features=dec_embed_dim, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.pre_mlp_norm = nnx.RMSNorm(num_features=dec_embed_dim, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.self_attention = Attention(config=config, q_embed_dim=dec_embed_dim, kv_embed_dim=dec_embed_dim, num_query_heads=dec_config.gqa_query_heads, num_kv_heads=dec_config.kv_heads, head_dim=dec_config.gqa_head_dim, dropout_rate=model_config.dropout, is_cross_attn=False, out_embed_dim=dec_embed_dim, rngs=rngs)
        self.cross_attention = Attention(config=config, q_embed_dim=dec_embed_dim, kv_embed_dim=enc_embed_dim,  num_query_heads=dec_config.cross_query_heads, num_kv_heads=dec_config.cross_query_heads, head_dim=dec_config.cross_head_dim, dropout_rate=model_config.dropout, is_cross_attn=True, out_embed_dim=dec_embed_dim, rngs=rngs)
        self.mlp = MlpBlock(config=config, embed_dim=dec_embed_dim, intermediate_dim=dec_config.n_hidden, use_pre_norm=dec_config.use_pre_norm, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x, self_attn_mask, cross_attn_mask, self_rope_cos, self_rope_sin, cross_rope_cos, cross_rope_sin, self_attn_cache, cross_attn_cache):
        residual = x
        x_norm = self.pre_sa_norm(x)
        sa_out, new_kv = self.self_attention(x=x_norm, rope_cos=self_rope_cos, rope_sin=self_rope_sin, attn_mask=self_attn_mask, kv_cache=self_attn_cache)
        x = residual + sa_out
        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out, _ = self.cross_attention(x=x_norm, rope_cos=cross_rope_cos, rope_sin=cross_rope_sin, attn_mask=cross_attn_mask, kv_cache=cross_attn_cache)
        x = residual + ca_out
        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x, new_kv

class Decoder(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        train_config = config.training
        data_config = config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer
        self.embeddings = [nnx.Embed(num_embeddings=model_config.tgt_vocab_size, features=dec_config.n_embd, rngs=rngs) for _ in range(self.num_channels)]
        self.layers = [DecoderLayer(config=config, rngs=rngs) for _ in range(self.num_layers)]
        self.norm = nnx.RMSNorm(num_features=dec_config.n_embd, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
        self.logits_dense = nnx.LinearGeneral(in_features=dec_config.n_embd, out_features=(self.num_channels, model_config.tgt_vocab_size), axis=-1, use_bias=False, rngs=rngs)
    
    # @nnx.jit
    def __call__(self, tgt_ids_Bx1xC, tgt_pos_Bx1, self_rope_cos, self_rope_sin, cross_rope_cos, cross_rope_sin, cross_attn_mask, self_kv_caches, cross_kv_caches):
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed
        new_self_kv_caches = [] # [] vmap
        for i, layer in enumerate(self.layers):
            x, new_kv = layer(x, self_attn_mask=None, cross_attn_mask=cross_attn_mask, self_rope_cos=self_rope_cos, self_rope_sin=self_rope_sin, cross_rope_cos=cross_rope_cos, cross_rope_sin=cross_rope_sin, self_attn_cache=self_kv_caches[i], cross_attn_cache=cross_kv_caches[i])
            new_self_kv_caches.append(new_kv)
        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)
        return logits_Bx1xCxV, new_self_kv_caches

    @nnx.jit
    def precompute_cross_attention_kv(self, max_len, encoder_out, rope_cos, rope_sin):
        cross_kv_caches = []
        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k = cross_attn_module.k_proj(encoder_out)
            v = cross_attn_module.v_proj(encoder_out)
            k = apply_rope(k, rope_cos, rope_sin)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))
            if cross_attn_module.num_gqa_groups > 1:
                k = jnp.repeat(k, cross_attn_module.num_gqa_groups, axis=1)
                v = jnp.repeat(v, cross_attn_module.num_gqa_groups, axis=1)
            cross_kv_caches.append((k, v))
        return cross_kv_caches
    
class DiaModel(nnx.Module):
    def __init__(self, config: DiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.encoder = Encoder(config, rngs=rngs)
        self.decoder = Decoder(config, rngs=rngs)

@nnx.jit
def sample_next_token(logits, top_p, cfg_filter_top_k, rng_key):
    sorted_indices = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    batch_size, vocab_size = logits.shape
    top_k_mask = jnp.arange(vocab_size) < cfg_filter_top_k
    top_k_mask = jnp.broadcast_to(top_k_mask, (batch_size, vocab_size))
    filtered_logits = jnp.where(top_k_mask, sorted_logits, -1e9)
    probs = jax.nn.softmax(filtered_logits, axis=-1)
    cumulative_probs = jnp.cumsum(probs, axis=-1)
    top_p_mask = cumulative_probs <= top_p
    top_p_mask = top_p_mask | (jnp.arange(vocab_size) == 0).reshape(1, -1)
    final_logits = jnp.where(top_p_mask, filtered_logits, -1e9)
    final_logits_original = jnp.zeros_like(logits) - 1e9
    final_logits_original = final_logits_original.at[jnp.arange(batch_size)[:, None], sorted_indices].set(final_logits)
    return jax.random.categorical(rng_key, final_logits_original)

def generate(model, config, text, max_tokens=None, cfg_scale=3.0, temperature=1.3, top_p=0.95, use_cfg_filter=True, cfg_filter_top_k=35, seed=0):
    key = jax.random.PRNGKey(seed)
    num_channels = config.data.channels
    audio_bos_value = config.data.audio_bos_value
    audio_eos_value = config.data.audio_eos_value
    audio_pad_value = config.data.audio_pad_value
    delay_pattern = config.data.delay_pattern
    if max_tokens is None:
        max_tokens = config.data.audio_length
    delay_tensor = jnp.array(delay_pattern, dtype=jnp.int32)
    max_delay_pattern = max(delay_pattern)
    byte_text = text.encode("utf-8")
    replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
    text_tokens = list(replaced_bytes)
    text_pad_value = config.data.text_pad_value
    max_len = config.data.text_length
    current_len = len(text_tokens)
    padding_needed = max_len - current_len
    if padding_needed <= 0:
        text_tokens = text_tokens[:max_len]
        padded_text_np = np.array(text_tokens, dtype=np.uint8)
    else:
        padded_text_np = np.pad(text_tokens, (0, padding_needed), mode="constant", constant_values=text_pad_value)
    cond_src_BxS = jnp.array(padded_text_np, dtype=jnp.int32)[None, :]  # [1, S]
    cond_src_positions_BxS = jnp.arange(max_len, dtype=jnp.int32)[None, :]  # [1, S]
    cond_src_padding_mask_BxS = (cond_src_BxS != text_pad_value)  # [1, S]
    unc_src_BxS = jnp.zeros_like(cond_src_BxS)
    src_BxS = jnp.concatenate([unc_src_BxS, cond_src_BxS], axis=0)  # [2, S]
    src_positions_BxS = jnp.broadcast_to(cond_src_positions_BxS, (2, cond_src_positions_BxS.shape[1]))
    src_padding_mask_BxS = jnp.broadcast_to(cond_src_padding_mask_BxS, (2, cond_src_padding_mask_BxS.shape[1]))
    enc_self_attn_mask = create_attention_mask(src_padding_mask_BxS, src_padding_mask_BxS)
    enc_roper = Roper(head_dim=config.model.encoder.head_dim, min_timescale=config.model.rope_min_timescale, max_timescale=config.model.rope_max_timescale)
    dec_self_roper = Roper(head_dim=config.model.decoder.gqa_head_dim, min_timescale=config.model.rope_min_timescale, max_timescale=config.model.rope_max_timescale)
    dec_cross_roper = Roper(head_dim=config.model.decoder.cross_head_dim, min_timescale=config.model.rope_min_timescale, max_timescale=config.model.rope_max_timescale)
    enc_rope_cos, enc_rope_sin = enc_roper(src_positions_BxS)
    encoder_out = model.encoder(x_ids=src_BxS, attn_mask=enc_self_attn_mask, rope_cos=enc_rope_cos, rope_sin=enc_rope_sin)
    self_kv_caches = [None]*len(model.decoder.layers)
    dec_cross_kv_cos, dec_cross_kv_sin = dec_cross_roper(src_positions_BxS)
    cross_kv_caches = model.decoder.precompute_cross_attention_kv(max_tokens, encoder_out, dec_cross_kv_cos, dec_cross_kv_sin)
    generated_BxTxC = jnp.full((2, max_tokens + 1, num_channels), -1, dtype=jnp.int32)
    generated_BxTxC = generated_BxTxC.at[:, 0, :].set(audio_bos_value)
    tgt_padding_mask = jnp.ones((2, 1), dtype=bool)
    decoder_cross_attn_mask = create_attention_mask(tgt_padding_mask, src_padding_mask_BxS)
    eos_detected_channel_0 = False
    eos_countdown = -1
    extra_steps_after_eos = 30
    V = config.model.tgt_vocab_size
    for step in range(400): # DEBUG
        key, subkey = jax.random.split(key)
        tgt_ids_Bx1xC = generated_BxTxC[:, step:step+1, :]
        tgt_pos_Bx1 = jnp.full((2, 1), fill_value=step, dtype=jnp.int32)
        self_rope_cos, self_rope_sin = dec_self_roper(tgt_pos_Bx1)
        cross_q_cos, cross_q_sin = dec_cross_roper(tgt_pos_Bx1)
        logits_Bx1xCxV, self_kv_caches = model.decoder(tgt_ids_Bx1xC=tgt_ids_Bx1xC, tgt_pos_Bx1=tgt_pos_Bx1, self_rope_cos=self_rope_cos, self_rope_sin=self_rope_sin, cross_rope_cos=cross_q_cos, cross_rope_sin=cross_q_sin, cross_attn_mask=decoder_cross_attn_mask, self_kv_caches=self_kv_caches, cross_kv_caches=cross_kv_caches)
        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :] 
        uncond_logits_CxV = logits_last_BxCxV[0] 
        cond_logits_CxV = logits_last_BxCxV[1]  
        cfg_logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        logits_CxV = cfg_logits_CxV.reshape((-1, V))
        logits_CxV = logits_CxV.at[:, 1025:].set(-1e9)  
        pred_C = sample_next_token(logits_CxV/temperature, top_p=top_p, cfg_filter_top_k=cfg_filter_top_k, rng_key=key) if temperature > 0 else jnp.argmax(logits_CxV, axis=-1)
        # print(f'{pred_C=}')
        generation_step_index = step
        delay_mask = generation_step_index >= delay_tensor
        pred_C = jnp.where(delay_mask, pred_C, audio_bos_value)
        generated_BxTxC = generated_BxTxC.at[:, step+1, :].set(jnp.broadcast_to(pred_C[None, :], (2, pred_C.shape[0])))
        if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
            eos_detected_channel_0 = True
            eos_countdown = extra_steps_after_eos
        if eos_countdown > 0:
            step_after_eos = max_delay_pattern - eos_countdown
            for i, d in enumerate(delay_pattern):
                if step_after_eos == d:
                    generated_BxTxC = generated_BxTxC.at[:, step+1, i].set(audio_eos_value)
                elif step_after_eos > d:
                    generated_BxTxC = generated_BxTxC.at[:, step+1, i].set(audio_pad_value)
            eos_countdown -= 1
            if eos_countdown == 0:
                break
    output_codes = generated_BxTxC[:, 1 : step + 1, :]
    generated_codes = output_codes[0]
    return codebook_to_audio(generated_codes.transpose(1, 0), delay_pattern, B=1, T=max_tokens, C=num_channels)

def load(model_name = 'jaco-bro/Dia-1.6B'):
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    checkpoint_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    config = DiaConfig.load(config_path)
    graphdef, state = nnx.split(nnx.eval_shape(lambda: DiaModel(config, rngs=nnx.Rngs(0))))
    state_dict = dict(state.flat_state())
    for path, val in ((k.replace('weight', 'embedding') if 'embeddings' in k else k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("wi_fused.weight", "wi_fused.kernel").replace("wo.weight", "wo.kernel").replace("embedding.weight", "embedding.embedding").replace('logits_dense.weight', 'logits_dense.kernel'), nnx.Param(jnp.array(v))) for k, v in load_file(checkpoint_path).items()):
        state_dict[tuple(int(part) if part.isdigit() else part for part in path.split('.'))].value = jnp.array(val, dtype=jnp.bfloat16)
    return nnx.merge(graphdef, nnx.State.from_flat_path(state_dict)), config

def main():
    import argparse
    import soundfile as sf
    parser = argparse.ArgumentParser(description='Dia-JAX: Generate dialogue audio from text')
    parser.add_argument('--text', type=str, default="[S1] Dear Jacks, to generate audio from text from any machine. (applause) [S2] Really? How! (screams) [S1] With flakes and an axe. (chuckles)",
                        help='Input text with [S1] and [S2] speaker tags'),
    parser.add_argument('--output', type=str, default='output.mp3',
                        help='Output audio filename')
    parser.add_argument('--model', type=str, default='jaco-bro/Dia-1.6B',
                        help='Model name or path')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--cfg-scale', type=float, default=3.0,
                        help='CFG scale for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generation')
    parser.add_argument('--no-cfg-filter', action='store_false', dest='use_cfg_filter',
                        help='Disable CFG filtering')
    parser.add_argument('--cfg-filter-top-k', type=int, default=35,
                        help='Top-k for CFG filtering')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model, config = load(args.model)
    
    print(f"Generating audio for text: {args.text}")
    output = generate(
        model, 
        config, 
        args.text,
        max_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cfg_filter=args.use_cfg_filter,
        cfg_filter_top_k=args.cfg_filter_top_k,
        seed=args.seed
    )
    
    del model
    print(f"Saving audio to {args.output}")
    sf.write(args.output, output, 44100)
    print(f"Done! Audio saved to {args.output}")

if __name__ == "__main__":
    main()
