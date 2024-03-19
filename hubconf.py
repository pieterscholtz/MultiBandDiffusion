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
    model = torch.hub.load('pieterscholtz/MultiBandDiffusion', output_sr=24000, bandwidth=6.0, device=device)

    # Input audio file
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.to('cuda')

    # Extract tokens
    tokens = model.codec_model.encode(waveform)
    tokens = tokens[0][0][0].detach()

    # Convert to continuous latent space
    condition = model.get_emb(tokens.unsqueeze(0))

    # Convert to waveform
    with torch.no_grad():
        wav_diffusion = model.generate(emb=condition)
    
    torchaudio.save('out.wav;, wav_diffusion.squeeze(0).cpu(), 24000)
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
