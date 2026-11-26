# UTO-Omega-Music v0.2 — built-in singing and studio DSP

Offline procedural original music, formant vocal synthesis, and optional FFmpeg-based studio processing. Research prototype, not a claim of human-quality singing. Python 3.10+ and NumPy required. No model weights, voice cloning, remote API, network access or paid service at runtime. The 801-byte plain-text launcher delegates to the included engines; the engines are not included in the 2KB count.

## Setup

Install Python and NumPy (`python3 -m pip install -r requirements.txt`). Install FFmpeg separately for optional mastering. Run from the repository directory or set UTO_MUSIC_HOME.

```sh
sh UTO-Omega-Music-2KB.sh --style cinematic --key D --bars 8 --seed 777 --out instrumental
sh UTO-Omega-Music-2KB.sh --voice synth --syllables "la lo mi na" --style cinematic --key D --bpm 96 --bars 4 --seed 777 --stems --out song
python3 studio.py song --lufs -14 --true-peak -1.5
sh UTO-Omega-Music-2KB.sh --voice wav --voice-wav singer.wav --authorized --bars 4 --voice-offset 0.5 --out recorded
```

Synth mode follows the generated melody with harmonic/formant synthesis, five vowel shapes, approximate consonant noise, breath, onset/release and delayed vibrato. Syllables repeat; no natural-language phonemizer or neural vocalist. Adjust --voice-octave, --vibrato (cents), --voice-gain, --voice-offset (seconds). WAV mode requires authorized 16/24-bit PCM mono/stereo matching the backing sample rate (48 kHz or --rate 44100), manual timing and truncation to backing duration. No automatic pitch/time correction or consent verification. Use only audio you have rights to.

Outputs: score.mid, mix.wav (instrumental), vocal.wav, vocal-mix.wav, session.json, optional instrumental stems. The basic mix uses backing * 0.7 plus vocal, then sample-peak gain. Studio mode remixes the vocal stem and backing independently with high-pass filtering, EQ, de-essing, compression and two-pass loudness normalization; outputs studio-master.wav (48 kHz, 24-bit stereo) and studio-report.json. Output targets are configurable and measured again after rendering; the reported within_target field is a measurement, not a quality certification. Existing output directories or studio outputs are not overwritten.

## Checks

```sh
sh UTO-Omega-Music-2KB.sh self | cmp - UTO-Omega-Music-2KB.sh
sh UTO-Omega-Music-2KB.sh decode "$(sh UTO-Omega-Music-2KB.sh encode)" | cmp - UTO-Omega-Music-2KB.sh
```

Local prototype checks: BusyBox ash self reproduction and G257 roundtrip; same-seed synthetic vocal mixes identical; an 11.2-second D-minor cinematic example produced measured -14.01 LUFS and -1.94 dBTP after studio processing (target -14 LUFS, -1.5 dBTP ceiling). Measurements do not imply perceptual quality. iSH and professional listening panel not tested. No CI or deployment.

## Roadmap

1. Better phoneme timing, coarticulation, expressive pitch and breath, independent vocal melody and harmony.
2. Optional locally run trained singing model with documented licensed/consented training data, phoneme alignment and a vocoder.
3. Listening tests and pitch, intelligibility, timing, noise and loudness audits; compare with independent professional productions.

Original instrumental engine supports ambient, cinematic, electronic and jazz; 1–128 bars (audio capped at 300 seconds before release), BPM 40–220. Neural singing and free-text multilingual lyrics are not implemented.
