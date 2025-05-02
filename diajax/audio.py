import typing as tp
import numpy as np
import torch
import torchaudio
import dac

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_delay_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)
    t_idx_BxT = torch.broadcast_to(torch.arange(T, dtype=torch.int32)[None, :], [B, T])
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, dtype=torch.int32).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, dtype=torch.int32).view(1, 1, C), [B, T, C])
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)
    indices_BTCx3 = torch.stack([b_idx_BxTxC.reshape(-1), t_clamped_BxTxC.reshape(-1), c_idx_BxTxC.reshape(-1)], dim=1).long()
    return t_idx_BxTxC, indices_BTCx3

def apply_audio_delay(audio_BxTxC: torch.Tensor, pad_value: int, bos_value: int, precomp: tp.Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    device = audio_BxTxC.device
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)
    mask_bos = t_idx_BxTxC < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))
    return result_BxTxC

@torch.no_grad()
@torch.inference_mode()
def audio_to_codebook(model, input_values, c, p, b, d, padding_mask=None, sample_rate=44100):
    audio_data = model.preprocess(input_values, sample_rate)
    if padding_mask is None:
        padding_mask = torch.ones_like(input_values).bool()
    _, encoded_frame, _, _, _ = model.encode(audio_data, n_quantizers=None)  # 1, C, T
    seq_length = encoded_frame.shape[2]
    t_idx_BxTxC, indices_BTCx3 = build_delay_indices(B=1, T=seq_length, C=c, delay_pattern=d)
    encoded_frame = apply_audio_delay(audio_BxTxC=encoded_frame.transpose(1, 2), pad_value=p, bos_value=b, precomp=(t_idx_BxTxC, indices_BTCx3))
    return encoded_frame

@torch.no_grad()
@torch.inference_mode()
def get_audio_prompt(audio_prompt_path, *, c=9, p=1025, b=1026, d=[0, 8, 9, 10, 11, 12, 13, 14, 15]):
    try:
        device = get_default_device()
        model = dac.DAC.load(dac.utils.download()).to(device)
        generated_BxTxC = torch.full((2, 1, c), fill_value=b, dtype=torch.long, device=device)
        audio_prompt, sr = torchaudio.load(audio_prompt_path, channels_first=True)  # C, T
        if sr != 44100:  # Resample to 44.1kHz
            audio_prompt = torchaudio.functional.resample(audio_prompt, sr, 44100)
        if audio_prompt.shape[0] > 1:
            audio_prompt = torch.mean(audio_prompt, dim=0, keepdim=True)  # Now shape is (1, T)
        audio_prompt = audio_prompt[:,:3*44100]
        audio_prompt = audio_prompt.to(device).unsqueeze(0)  # 1, C, T
        audio_prompt = audio_to_codebook(model, audio_prompt, c, p, b, d)
        generated_BxTxC = torch.cat([generated_BxTxC, audio_prompt.expand(2, -1, -1)], dim=1)
        return generated_BxTxC.to(torch.int32).cpu().numpy()
    except Exception as e:
        print(f'Error in encode: {e}')
        return np.full((2, 1, c), b, dtype=np.int32)

@torch.no_grad()
@torch.inference_mode()
def get_audio_values(codebook):
    try:
        device = get_default_device()
        model = dac.DAC.load(dac.utils.download()).to(device)
        codebook_torch = torch.from_numpy(np.array(codebook)).long().to(device)
        audio_values = model.quantizer.from_codes(codebook_torch)
        audio_values = model.decode(audio_values[0])
        return audio_values.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error in decode: {e}")
        return np.zeros((1, 44100))
