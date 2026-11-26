# UTO-Omega-ZPF v0.1 — paired classical field experiment

ZPF-inspired DSP, not a quantum field simulation, physical zero-point energy extraction, or evidence for an improvement in music quality. All energy for computation and playback is supplied by ordinary hardware. The existing music, neural and studio engines are unchanged.

The finite classical random-phase field is Z(t) = sum_k sqrt(2 f_k / sum_j f_j) cos(2 pi f_k t + phi_k). The mode-power weights are proportional to frequency, inspired by the single oscillator ground-state energy hbar omega/2; this is NOT the spectral density of the electromagnetic vacuum. Independent uniform pseudorandom phases give ensemble mean zero and ensemble variance one at any fixed time. The resulting finite realization need not have exact unit temporal RMS. Two bands: 24 linearly spaced modes from 0.5 to 30 Hz for pitch control and 48 modes from 400 to 4000 Hz for breath. Added pitch cents = 8 * strength * tanh(Z). Added breath coefficient = 0.003 * strength. Strength 0..1, default 0.75. No claim of special frequencies, healing or quantum randomness.

## Reproduce

Install Python, NumPy, PyTorch, Git and optional FFmpeg separately. Review the launcher and source before running. The 1007-byte launcher fetches a fixed Git revision into a temporary directory, then runs locally. Network fetch not tested in the local sandbox. Offline mode: set UTO_MUSIC_HOME to the reviewed checkout with zpf.py, neural.py, music.py and studio.py. Git commit ID is a revision pin, not a custom SHA algorithm. No automatic installations, paid API, cloud execution or upload of audio.

    sh UTO-Omega-ZPF-2KB.sh --bars 4 --bpm 96 --seed 777 --strength 0.75 --stems --master --out zpf-song
    sh UTO-Omega-ZPF-2KB.sh --model zpf-song/neural-weights.npz --strength 0 --out zero-test

The output path must not exist. Default renders a paired baseline and field experiment; --master runs the existing studio module independently on each. Both variants have the same composition, instrumental PCM, model weights, original seeded breath and vibrato, and shared gain before mastering. Each directory has a vocal stem and vocal mix, backing and MIDI; optional stems and masters. The report records settings, training, waveform difference and mastering measurements. Separate master normalization can change relative gain, so use raw paired vocal-mix.wav for isolation.

## Local test

Python CPU, NumPy 2.3.5, PyTorch 2.10.0+cpu, BusyBox ash and FFmpeg. Baseline and field: 11.2 s, 48000 Hz, 24-bit stereo, D minor cinematic, 96 BPM, 4 bars, seed 777, strength 0.75, 180 synthetic training updates. Added pitch bound 6 cents. Baseline vocal RMS 0.0843173, vocal waveform difference RMS 0.0202866 (linear full scale, before shared gain; not a quality metric). Masters: baseline -14.0 LUFS / -3.21 dBTP; field -14.0 LUFS / -2.99 dBTP. Both passed the specified -14 LUFS, -1.5 dBTP ceiling targets. Model reload reproduced the paired vocal, mix, master and score files byte for byte in this environment. Strength zero produced identical baseline and field vocal and vocal-mix files. Backing and MIDI matched in the nonzero pair. Launcher one line, 1007 ASCII bytes, ash syntax, exact self reproduction and G257 roundtrip passed. Neural model still learns synthetic harmonic targets only; no human voice training, lyrics or professional listening panel. Blind A/B and ordinary colored-noise controls are needed to evaluate benefit. No CI, deployment or spending.
