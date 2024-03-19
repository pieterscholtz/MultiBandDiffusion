dependencies = ['torch', 'torchaudio', 'numpy', 'einops', 'encodec', 'omegaconf', 'julius', 'huggingface_hub']

import typing as tp
from pathlib import Path
from encodec import EncodecModel

import omegaconf

import torch
import torchaudio
import julius

from multibanddiffusion import MultiBandDiffusion
from unet import DiffusionUnet
from diffusion_schedule import NoiseSchedule, SampleProcessor, MultiBandProcessor


def mbd(output_sr=24000, bandwidth=6.0, device=None) -> MultiBandDiffusion:
    """ Loads 24kHz MultiBandDiffusion to decode EnCodec tokens with bandwidth of 6.0. 

    Example usage:
    ```python
    device = torch.device('cuda')
    model = torch.hub.load('pieterscholtz/MultiBandDiffusion', 'mbd', output_sr=24000, bandwidth=6.0, device=device)

    # Input audio file
    file = 'test/ex04_default_00364.wav'
    waveform, sample_rate = torchaudio.load(file)
    assert sample_rate == model.sample_rate, f'sample rate of {sample_rate} Hz is not supported'
    waveform.unsqueeze_(0)
    waveform = waveform.to(device)

    with torch.inference_mode():
        # Extract tokens
        tokens = model.codec_model.encode(waveform)
        tokens = tokens[0][0]
        # Convert to continuous latent space
        condition = model.get_emb(tokens)
        # Convert to waveform
        wav_diffusion = model.generate(emb=condition)

    out_wav = Path(file).with_suffix('.mbd.wav')
    torchaudio.save(out_wav, wav_diffusion.squeeze(0).cpu(), model.sample_rate)
    ```
    """
    assert output_sr in [24000], "output_sr argument must be 24000."
    assert bandwidth in [6.0], "bandwidth argument be 6.0."

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = MultiBandDiffusion.get_mbd_24khz(device=device)
    # print(f"Loaded MultiBandDiffusion with {sum(param.numel() for param in mbd.parameters()):,d} parameters.")
    return model
