import functools
import os
import time
from typing import List, Dict, Union, Optional, Tuple, Callable, Any
from pydantic import BaseModel, Field
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from . import audio
# import audio # DEBUG

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
    sow_len: int = Field(default=1024) # ad hoc
    kv_heads: int = Field(gt=0)
    gqa_head_dim: int = Field(gt=0)
    cross_query_heads: int = Field(gt=0)
    cross_head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)

class ModelConfig(BaseModel):
    encoder: EncoderConfig
    decoder: DecoderConfig
    src_vocab_size: int = Field(default=256, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    weight_dtype: str = Field(default="bfloat16")
    rope_min_timescale: int = Field(default=1)
    rope_max_timescale: int = Field(default=10_000)

class TrainingConfig(BaseModel):
    dtype: str = Field(default="bfloat16")
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
    return np.array(codebook)

def save(codebook, filename='output.mp3', sr=44100):
    import soundfile as sf
    output = audio.get_audio_values(codebook)
    sf.write(filename, output, sr)

def load(model_name='jaco-bro/Dia-1.6B', dtype=jnp.bfloat16):
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    checkpoint_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    config = DiaConfig.load(config_path)
    graphdef, state = nnx.split(nnx.eval_shape(lambda: DiaModel(config, rngs=nnx.Rngs(0))))
    state_dict = dict(state.flat_state())
    for path, val in ((k.replace('weight', 'embedding') if 'embeddings' in k else k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("wi_fused.weight", "wi_fused.kernel").replace("wo.weight", "wo.kernel").replace("embedding.weight", "embedding.embedding").replace('logits_dense.weight', 'logits_dense.kernel'), nnx.Param(jnp.array(v))) for k, v in load_file(checkpoint_path).items()):
        state_dict[tuple(int(part) if part.isdigit() else part for part in path.split('.'))].value = jnp.array(val, dtype=jnp.bfloat16)
    model = nnx.merge(graphdef, nnx.State.from_flat_path(state_dict))
    model.set_attributes(dtype=dtype, param_dtype=dtype)
    return model

@nnx.jit
def sample_next_token(temperature, top_p, cfg_filter_top_k, logits, rng_key):
    logits = logits/temperature
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

@nnx.jit
def apply_rope(k, cos, sin):
    k1, k2 = jnp.split(k, 2, axis=-1)
    return jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1).astype(jnp.bfloat16)

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
    return mask[:, None, :, :].astype(jnp.bfloat16)

def create_causal_mask(q_padding_mask, k_padding_mask, s):
    B1, Tq = q_padding_mask.shape
    B2, Tk = k_padding_mask.shape
    _Tk = jnp.maximum(s-Tk, 0)
    q_mask_BxTqx1 = q_padding_mask[:, :, None]
    k_mask_Bx1xTk = k_padding_mask[:, None, :]
    non_pad_attends_non_pad = q_mask_BxTqx1 & k_mask_Bx1xTk
    pad_attends_pad = (~q_mask_BxTqx1) & (~k_mask_Bx1xTk)
    mask = non_pad_attends_non_pad | pad_attends_pad
    causal_mask = jnp.tril(jnp.ones((Tq, Tk), dtype=bool))
    mask = mask & causal_mask
    mask = jnp.where(mask, 0.0, -1e9)
    mask = jnp.pad(mask, ((0,0),(0,0),(_Tk,0)), constant_values=-1e9)
    return mask[:, None, :, -s:].astype(jnp.bfloat16)

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

class BufCache(nnx.Module):
    def __init__(self, num_heads, max_len, head_dim, k=None, v=None):
        self.max_len = max_len
        self.k = nnx.Variable(jnp.zeros((2, num_heads, max_len, head_dim), dtype=jnp.bfloat16)) if k is None else nnx.Variable(k)
        self.v = nnx.Variable(jnp.zeros((2, num_heads, max_len, head_dim), dtype=jnp.bfloat16)) if v is None else nnx.Variable(v)

    def __call__(self, k, v):
        self.k.value = jnp.concat([self.k.value, k], axis=2)[:,:,-self.max_len:,:]
        self.v.value = jnp.concat([self.v.value, v], axis=2)[:,:,-self.max_len:,:]
        return self.k.value, self.v.value

class NoCache(nnx.Module):
    def __init__(self): pass
    def __call__(self, k, v):
        return k, v

class Attention(nnx.Module):
    def __init__(self, config, q_embed_dim, kv_embed_dim, num_query_heads, num_kv_heads, head_dim, dropout_rate, is_cross_attn=False, out_embed_dim=None, *, sow_len=1024, rngs: nnx.Rngs):
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
        self.window_size = sow_len
    
    @nnx.jit
    def __call__(self, x, rope_cos, rope_sin, attn_mask, kv_cache=None):
        q = self.q_proj(x)
        q = apply_rope(q, rope_cos, rope_sin)
        q = jnp.transpose(q, (0, 2, 1, 3))
        if self.is_cross_attn:
            k, v = kv_cache
        else:
            _k = self.k_proj(x)
            _v = self.v_proj(x)
            _k = apply_rope(_k, rope_cos, rope_sin)
            _k = jnp.transpose(_k, (0, 2, 1, 3))
            _v = jnp.transpose(_v, (0, 2, 1, 3))
            k, v = kv_cache(_k, _v)
            if self.num_gqa_groups > 1:
                k = jnp.repeat(k, self.num_gqa_groups, axis=1)
                v = jnp.repeat(v, self.num_gqa_groups, axis=1)
        w = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        w = w + attn_mask
        w = nnx.softmax(w, axis=-1)
        w = jnp.matmul(w, v)
        w = jnp.transpose(w, (0, 2, 1, 3))
        output = self.o_proj(w)
        return output

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
        sa_out = self.self_attention(x=self.pre_sa_norm(x), rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=attn_mask, kv_cache=NoCache())  
        x = residual + sa_out
        residual = x
        return residual + self.mlp(self.post_sa_norm(x))
        return x

class Encoder(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        model_config = config.model
        enc_config = config.model.encoder
        self.embedding = nnx.Embed(num_embeddings=model_config.src_vocab_size, features=enc_config.n_embd, rngs=rngs)
        self.layers = [EncoderLayer(config=config, rngs=rngs) for _ in range(enc_config.n_layer)]
        self.norm = nnx.RMSNorm(num_features=enc_config.n_embd, epsilon=model_config.normalization_layer_epsilon, rngs=rngs)
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
        self.self_attention = Attention(config=config, q_embed_dim=dec_embed_dim, kv_embed_dim=dec_embed_dim, num_query_heads=dec_config.gqa_query_heads, num_kv_heads=dec_config.kv_heads, head_dim=dec_config.gqa_head_dim, dropout_rate=model_config.dropout, is_cross_attn=False, out_embed_dim=dec_embed_dim, sow_len=dec_config.sow_len, rngs=rngs)
        self.cross_attention = Attention(config=config, q_embed_dim=dec_embed_dim, kv_embed_dim=enc_embed_dim,  num_query_heads=dec_config.cross_query_heads, num_kv_heads=dec_config.cross_query_heads, head_dim=dec_config.cross_head_dim, dropout_rate=model_config.dropout, is_cross_attn=True, out_embed_dim=dec_embed_dim, rngs=rngs)
        self.mlp = MlpBlock(config=config, embed_dim=dec_embed_dim, intermediate_dim=dec_config.n_hidden, use_pre_norm=dec_config.use_pre_norm, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x, self_attn_mask, cross_attn_mask, self_rope_cos, self_rope_sin, cross_rope_cos, cross_rope_sin, cross_attn_cache, self_attn_cache):
        residual = x
        x_norm = self.pre_sa_norm(x)
        sa_out = self.self_attention(x=x_norm, rope_cos=self_rope_cos, rope_sin=self_rope_sin, attn_mask=self_attn_mask, kv_cache=self_attn_cache)
        x = residual + sa_out
        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.cross_attention(x=x_norm, rope_cos=cross_rope_cos, rope_sin=cross_rope_sin, attn_mask=cross_attn_mask, kv_cache=cross_attn_cache)
        x = residual + ca_out
        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x

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
    
    @nnx.jit
    def __call__(self, cross_attn_mask, cross_kv_caches, tgt_ids_Bx1xC, self_rope_cos, self_rope_sin, cross_rope_cos, cross_rope_sin, self_attn_mask, self_attn_caches):
        x = sum(emb(tgt_ids_Bx1xC[..., i]) for i, emb in enumerate(self.embeddings))
        for i, layer in enumerate(self.layers):
            x = layer(x, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask, self_rope_cos=self_rope_cos, self_rope_sin=self_rope_sin, cross_rope_cos=cross_rope_cos, cross_rope_sin=cross_rope_sin, cross_attn_cache=cross_kv_caches[i], self_attn_cache=self_attn_caches[i])
        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)
        return logits_Bx1xCxV

    @nnx.jit
    def precompute_cross_attention_kv(self, max_len, encoder_out, rope_cos, rope_sin):
        def single_layer_kv(layer):
            l = layer.cross_attention
            k = l.k_proj(encoder_out)
            v = l.v_proj(encoder_out)
            k = apply_rope(k, rope_cos, rope_sin)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))
            k = jnp.repeat(k, l.num_gqa_groups, axis=1)
            v = jnp.repeat(v, l.num_gqa_groups, axis=1)
            return jnp.stack([k, v])
        return jax.tree.map(single_layer_kv, self.layers)

class DiaModel(nnx.Module):
    def __init__(self, config: DiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.encoder = Encoder(config, rngs=rngs)
        self.decoder = Decoder(config, rngs=rngs)

class Carry(nnx.Module):
    def __init__(self, tok, idx, caches, penalty):
        self.set(tok, idx, caches, penalty)
    def __call__(self):
        return self.tok.value, self.idx.value, self.caches, self.penalty.value
    def set(self, tok, idx, caches, penalty):
        self.tok=nnx.Variable(tok)
        self.idx=nnx.Variable(jnp.array(idx, dtype=jnp.int32))
        self.caches=caches
        self.penalty=nnx.Variable(jnp.array(penalty, dtype=jnp.bfloat16))

def generate(model, text, audio_prompt=None, max_tokens=None, cfg_scale=3.0, temperature=0.7, top_p=0.95, use_cfg_filter=True, cfg_filter_top_k=35, seed=0, scan_batch_size=500, use_scan=False):
    tic = time.perf_counter()
    config = model.config
    key = jax.random.PRNGKey(seed)
    num_channels = config.data.channels
    audio_bos_value = config.data.audio_bos_value
    audio_eos_value = config.data.audio_eos_value
    audio_pad_value = config.data.audio_pad_value
    delay_pattern = config.data.delay_pattern
    max_tokens = config.data.audio_length if max_tokens is None else min(max_tokens, config.data.audio_length)
    delay_tensor = jnp.array(delay_pattern, dtype=jnp.int32)
    max_delay_pattern = max(delay_pattern)
    byte_text = text.encode("utf-8")
    replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
    text_tokens = list(replaced_bytes)
    text_pad_value = config.data.text_pad_value
    max_len = config.data.text_length
    current_len = len(text_tokens)
    padding_needed = np.maximum(0, max_len - len(text_tokens))
    text_tokens = text_tokens[:max_len]
    padded_text_np = np.pad(text_tokens, (0, padding_needed), mode="constant", constant_values=text_pad_value).astype(np.uint8)
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
    dec_cross_kv_cos, dec_cross_kv_sin = dec_cross_roper(src_positions_BxS)
    cross_kv_caches = model.decoder.precompute_cross_attention_kv(max_tokens, encoder_out, dec_cross_kv_cos, dec_cross_kv_sin)
    tok = jnp.full((2, 1, num_channels), fill_value=audio_bos_value, dtype=jnp.int32) if audio_prompt is None else audio_prompt
    prompt_len_inc_bos = prefill_len = tok.shape[1]
    prefill_tgt_padding_mask = jnp.any(tok != audio_pad_value, axis=2)
    sow_len = config.model.decoder.sow_len
    self_attn_caches = [BufCache(4, sow_len, 128) for _ in range(config.model.decoder.n_layer)] # !!!
    prefill_self_attn_mask = create_causal_mask(prefill_tgt_padding_mask, prefill_tgt_padding_mask, sow_len)
    prefill_cross_attn_mask = create_attention_mask(prefill_tgt_padding_mask, src_padding_mask_BxS)
    prefill_tgt_pos = jnp.broadcast_to(jnp.arange(prefill_len, dtype=jnp.int32)[None, :], (2, prefill_len))
    prefill_self_cos, prefill_self_sin = dec_self_roper(prefill_tgt_pos)
    prefill_cross_cos, prefill_cross_sin = dec_cross_roper(prefill_tgt_pos)
    model.decoder(prefill_cross_attn_mask, cross_kv_caches, tok, prefill_self_cos, prefill_self_sin, prefill_cross_cos, prefill_cross_sin, prefill_self_attn_mask, self_attn_caches)
    current_step = prompt_len_inc_bos - 1
    tgt_padding_mask = jnp.ones((2, 1), dtype=bool)
    decoder_cross_attn_mask = create_attention_mask(tgt_padding_mask, src_padding_mask_BxS)
    eos_detected_channel_0 = False
    eos_countdown = -1
    extra_steps_after_eos = 30
    # _sample = functools.partial(sample_next_token, temperature, top_p, cfg_filter_top_k) if temperature > 0.01 else lambda x,_:jnp.argmax(x, axis=-1)
    V = config.model.tgt_vocab_size
    # carry = Carry(tok[:, :1, :], jnp.array(current_step, dtype=jnp.int32), self_attn_caches, jnp.array(0.0, dtype=jnp.bfloat16))
    # carry = Carry(tok[:, -1:, :], jnp.array(current_step, dtype=jnp.int32), self_attn_caches, jnp.array(0.0, dtype=jnp.bfloat16))
    # c0, carry_1 = nnx.split(carry)
    carry_1 = (tok[:, -1:, :], jnp.array(current_step, dtype=jnp.int32), self_attn_caches, jnp.array(0.0, dtype=jnp.bfloat16))
    # @nnx.jit
    def decode_fn(c1, key):
        # _carry = nnx.merge(c0, c1)
        # tok, idx, caches, penalty = _carry()
        tok, idx, caches, penalty = c1
        step = idx + 1
        g_step = idx-current_step
        # pos = jnp.full((2, 1), g_step, dtype=jnp.int32)
        pos = jnp.full((2, 1), idx, dtype=jnp.int32)
        self_cos, self_sin = dec_self_roper(pos)
        cross_cos, cross_sin = dec_cross_roper(pos)
        buf_mask = jnp.arange(sow_len) >= (sow_len - step)
        logits_B1CxV = model.decoder(decoder_cross_attn_mask, cross_kv_caches, tok, self_cos, self_sin, cross_cos, cross_sin, jnp.where(buf_mask, 0, -1e9).astype(jnp.bfloat16), caches)
        logits_last = logits_B1CxV[:, -1, :, :]
        uncond, cond = logits_last[0], logits_last[1]
        cfg_logits = cond + cfg_scale * (cond - uncond)
        logits_CV = cfg_logits[:,:1025]
        pred_C = sample_next_token(temperature, top_p, cfg_filter_top_k, logits_CV/(1.0+penalty), key)
        delay_mask = idx >= delay_tensor
        pred_C = jnp.where(delay_mask, pred_C, audio_bos_value)
        next_tok = jnp.broadcast_to(pred_C[None, None, :], (2, 1, num_channels))
        rep_penalty = jnp.all(tok == next_tok).astype(jnp.bfloat16)
        # _carry.set(next_tok, step, caches, rep_penalty)
        # _, _c1 = nnx.split(_carry)
        # return _c1, pred_C
        return (next_tok, step, caches, rep_penalty), pred_C
    # _decode_fn = functools.partial(decode_fn, model, carry_0)
    # _decode_fn = nnx.cached_partial(decode_fn, model, carry_0)
    result = []
    if use_scan:
        scan_fn = nnx.scan(decode_fn, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    else:
        _decode_fn = nnx.jit(decode_fn)
        def not_scan(carry, keys):
            outputs = []
            for key in keys:
                carry, output = _decode_fn(carry, key)
                outputs.append(output)
                if output[0] == audio_eos_value:
                    break
            stacked_outputs = jnp.stack(outputs, axis=0) if outputs else jnp.array([])
            return carry, stacked_outputs
        scan_fn = not_scan
    benchmark_enc = time.perf_counter() - tic
    tic = time.perf_counter()
    remaining = max_tokens
    while remaining > 0:
        current_batch_size = min(scan_batch_size, remaining)
        keys = jax.random.split(key, current_batch_size + 1)
        carry_1, outputs = scan_fn(carry_1, keys[:-1])
        eos_indices = jnp.where(outputs[:, 0] == audio_eos_value)[0]
        if eos_indices.size > 0:
            result.append(outputs[:eos_indices[0]+1])
            break
        else:
            result.append(outputs)
            key = keys[-1]
            remaining -= current_batch_size
    benchmark_time = time.perf_counter() - tic
    result = jnp.concat(result).T
    benchmark_step = result.size/9
    benchmark_tps = benchmark_step / benchmark_time
    print(f'{benchmark_enc:.2f} seconds in starting up\n{benchmark_tps:.2f} tokens-per-second ({benchmark_step} tokens in {benchmark_time:.2f} seconds)')
    return codebook_to_audio(result, delay_pattern, B=1, T=max_tokens, C=num_channels)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dia-JAX: Generate dialogue audio from text')
    parser.add_argument('--text', type=str, default="[S1] Dear Jacks, to generate audio from text from any machine. [S2] Any machine? (laughs) How? [S1] With flacks and an axe. (coughs) ",
    # parser.add_argument('--text', type=str, default="Hey, this is Don from the small winner club. Sorry to get back to you so late. I just finished reviewing your application and information you sent in. But I'm sorry to say that I don't think I can allow you to join our group. From what I'm looking at, (coughs) your win is just so massive it's ridiculous really.",
                        help='Input text with [S1] and [S2] speaker tags'),
    parser.add_argument('--audio', type=str, default=None, 
    # parser.add_argument('--audio', type=str, default='assets/example_prompt.mp3',
                        help='Input audio prompt filename to voice clone)')
    parser.add_argument('--output', type=str, default='output.mp3',
                        help='Output audio filename')
    parser.add_argument('--model', type=str, default='jaco-bro/Dia-1.6B',
                        help='Model name or path')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--cfg-scale', type=float, default=3.0,
                        help='CFG scale for generation')
    parser.add_argument('--temperature', type=float, default=1.3,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generation')
    parser.add_argument('--no-cfg-filter', action='store_false', dest='use_cfg_filter',
                        help='Disable CFG filtering')
    parser.add_argument('--cfg-filter-top-k', type=int, default=35,
                        help='Top-k for CFG filtering')
    parser.add_argument('--scan-batch-size', type=int, default=100,
                        help='Bigger the faster but beware the crash')
    parser.add_argument('--use-scan', action='store_true', dest='use_scan',
                        help='Enable flax.nnx.scan (Dont try on mac; jax-metal is broken)')
    args = parser.parse_args()
    audio_prompt = jnp.array(audio.get_audio_prompt(args.audio), dtype=jnp.int32) if args.audio is not None else None
    print(f"Loading model from {args.model}...")
    model = load(args.model)
    print(f"Generating audio for text: {args.text}")
    output = generate(
        model, 
        args.text,
        audio_prompt=audio_prompt,
        max_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cfg_filter=args.use_cfg_filter,
        cfg_filter_top_k=args.cfg_filter_top_k,
        seed=args.seed,
        scan_batch_size=args.scan_batch_size,
        use_scan=args.use_scan,
    )
    del model
    print(f"Audio saved to {args.output}")
    save(output, args.output)

if __name__ == "__main__":
    main()
