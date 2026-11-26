# UTO-Omega-Music Studio One-Liner v0.3

1767 ASCII bytes including the final newline. One command generates original procedural accompaniment, formant-synthesized vocals (or mixes authorized recorded vocals), MIDI, optional stems, and a measured stereo master. It is not a neural singing model or a guarantee of professional perceptual quality.

Requirements: Python 3.10+, NumPy (see requirements.txt), FFmpeg, Git and a network connection for remote mode. Install prerequisites separately. Review the launcher and pinned engine source before executing. No automatic package installation, credentials, paid API, or GitHub Actions workflow.

From any local directory, after saving UTO-Omega-Music-Studio-2KB.sh:
```sh
sh UTO-Omega-Music-Studio-2KB.sh --style cinematic --key D --bpm 96 --bars 4 --seed 777 --syllables "la lo mi na" --stems --out song
```
The default is synthesized vocals. Output directory must not exist. For offline use, set UTO_MUSIC_HOME to a reviewed local checkout containing music.py, voice.py and studio.py. Remote mode fetches the pinned commit 097e3f427bb9ace540e69d1af827b62eb22d04ea into a temporary directory; no tracking of moving main, no upload of audio. The commit identifier is a Git revision pin, not a custom SHA computation in the launcher.

Outputs include studio-master.wav, studio-report.json, vocal.wav, vocal-mix.wav, mix.wav, score.mid, session.json, and optional instrument stems. Target defaults: -14 LUFS, -1.5 dBTP ceiling; override with --lufs or --true-peak. Check within_target and listen to the result. The engine uses a restricted syllable set, not natural language lyrics. No claim of human-quality vocals.

```sh
sh UTO-Omega-Music-Studio-2KB.sh self | cmp - UTO-Omega-Music-Studio-2KB.sh
sh UTO-Omega-Music-Studio-2KB.sh decode "$(sh UTO-Omega-Music-Studio-2KB.sh encode)" | cmp - UTO-Omega-Music-Studio-2KB.sh
```
Local checks: BusyBox ash syntax, exact self reproduction and G257 roundtrip passed; the local engine files were compared with the archived GitHub blobs. The example produced 11.2 seconds at 48 kHz, 24-bit stereo, -14.01 LUFS and -1.94 dBTP, within target. The actual GitHub network checkout could not be exercised in the local test environment (DNS unavailable); it remains untested. No cloud execution or deployment was enabled. Existing v0.2 launcher and engines are unchanged.
