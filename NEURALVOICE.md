# UTO-NeuralVoice v0.1

Experimental CPU-trained neural harmonic envelope estimator plus additive DSP synthesis, integrated with music.py. It is not a neural waveform vocoder, a human-trained singing voice, or a professional perceptual-quality result. No recordings or pretrained voice weights are used. Only five vowel tokens (a e i o u) are supported. No consonant or free-text lyric synthesis in this module.

Install Python, NumPy and PyTorch using their trusted distribution instructions (no automatic installations). From a reviewed local checkout:

```sh
sh UTO-NeuralVoice-2KB.sh --bars 4 --bpm 96 --seed 777 --vowels "a e i o u" --expression 0.6 --vibrato 25 --stems --out neural-song
sh UTO-NeuralVoice-2KB.sh --model neural-song/neural-weights.npz --out neural-reload
python3 studio.py neural-song
```

The optional studio command requires FFmpeg and is a separate mastering step. Output directories must be new. Weights are generated locally or loaded from a compatible local NPZ (pickle disabled); no pretrained weights are bundled. Model has 4304 parameters, trained on 2048 synthetic examples and evaluated on 512 separate synthetic examples. Local CPU test at seed 777, 180 updates: validation MSE 0.007447557989507914 before and 0.000287248840322718 after. Reloaded weights produced byte-identical vocal.wav and vocal-mix.wav under the same environment and settings. Generated audio: 11.2 s, 48 kHz, 24-bit stereo. The 716-byte launcher passed BusyBox ash self-reproduction and G257 roundtrip. No network fetch in this launcher; clone the repository separately. Existing studio remote launcher still pins the earlier engine. No CI, cloud run, deployment or paid service enabled. Human voice training, phoneme alignment, prosody and listening evaluation remain future work.
