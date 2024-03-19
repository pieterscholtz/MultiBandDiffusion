# MultiBandDiffusion


## Minimal coding/decoding example (TODO: move to a notebook)

```python
    from multibanddiffusion import MultiBandDiffusion
    import torchaudio

    device = torch.device('cuda')
    model = MultiBandDiffusion.get_mbd_24khz(device)

    file = 'test/ex04_default_00364.wav'
    waveform, sample_rate = torchaudio.load(file)
    assert sample_rate == model.sample_rate, f'sample rate of {sample_rate} Hz is not supported'
    waveform.unsqueeze_(0)
    waveform = waveform.to(device)

    with torch.inference_mode():
        tokens = model.codec_model.encode(waveform)
        tokens = tokens[0][0]
        condition = model.get_emb(tokens)
        wav_diffusion = model.generate(emb=condition)

    out_wav = Path(file).with_suffix('.mbd.wav')
    torchaudio.save(out_wav, wav_diffusion.squeeze(0).cpu(), model.sample_rate)
```
