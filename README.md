# MultiBandDiffusion


## Minimal coding/decoding example (TODO: move to a notebook)

```python
    from multibanddiffusion import MultiBandDiffusion
    import torchaudio
    
    model = MultiBandDiffusion.get_mbd_24khz(bw=6.0)
    
    file = '/home/ubuntu/metavoice-src/inputs/commentary2.24kHz.wav'
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.to('cuda')
    
    tokens = model.codec_model.encode(waveform)
    tokens = tokens[0][0][0].detach()
    
    condition = model.get_emb(tokens.unsqueeze(0))
    
    with torch.no_grad():
        wav_diffusion = model.generate(emb=condition)
    
    out_wav = Path('.', Path('temp.wav').name)
    torchaudio.save(out_wav, wav_diffusion.squeeze(0).cpu(), 24000)
```
