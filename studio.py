"""Optional offline vocal mixing and two-pass EBU R128 mastering via FFmpeg."""
import argparse, json, math, re, shutil, subprocess, tempfile
from pathlib import Path


def run(ff, args):
    p = subprocess.run([ff, '-nostdin', '-hide_banner', '-v', 'info', *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return p.stderr


def loudness(ff, path, target):
    log = run(ff, ['-i', str(path), '-af', target+':print_format=json', '-f', 'null', '-'])
    data = json.loads(re.findall(r'\{[^{}]*\}', log)[-1])
    for key in ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset'):
        if not math.isfinite(float(data[key])):
            raise ValueError('Cannot normalize silent or unmeasurable material')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--lufs', type=float, default=-14)
    p.add_argument('--true-peak', type=float, default=-1.5)
    p.add_argument('--vocal-db', type=float, default=0)
    p.add_argument('--backing-db', type=float, default=-3)
    a = p.parse_args()
    if not (-24 <= a.lufs <= -9 and -6 <= a.true_peak <= -1 and -18 <= a.vocal_db <= 12 and -18 <= a.backing_db <= 3):
        p.error('target or gain out of supported range')
    ff = shutil.which('ffmpeg')
    if not ff:
        p.error('Install FFmpeg separately; no automatic downloads')
    dest = a.session/'studio-master.wav'; report = a.session/'studio-report.json'
    if dest.exists() or report.exists():
        p.error('Studio outputs already exist; will not overwrite')
    target = f'loudnorm=I={a.lufs}:TP={a.true_peak}:LRA=11'
    filters = ('[0:a]aformat=channel_layouts=stereo,highpass=f=80,'
               'equalizer=f=280:t=q:w=1:g=-2,equalizer=f=3200:t=q:w=1:g=1.5,'
               'deesser=i=0.25:m=0.35:f=0.5,acompressor=threshold=0.1:ratio=3:attack=15:release=120,'
               f'volume={a.vocal_db}dB[v];[1:a]aformat=channel_layouts=stereo,volume={a.backing_db}dB[b];'
               '[v][b]amix=inputs=2:duration=longest:normalize=0[m]')
    with tempfile.TemporaryDirectory() as tmp:
        mix = Path(tmp)/'mix.wav'; master = Path(tmp)/'master.wav'
        run(ff, ['-i', str(a.session/'vocal.wav'), '-i', str(a.session/'mix.wav'), '-filter_complex', filters, '-map', '[m]', '-ar', '48000', '-c:a', 'pcm_f32le', str(mix)])
        before = loudness(ff, mix, target)
        second = target + ''.join(':'+k+'='+before[v] for k,v in [('measured_I','input_i'),('measured_TP','input_tp'),('measured_LRA','input_lra'),('measured_thresh','input_thresh'),('offset','target_offset')]) + ':linear=true'
        run(ff, ['-i', str(mix), '-af', second, '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s24le', str(master)])
        after = loudness(ff, master, target)
        data = {'target_lufs': a.lufs, 'target_dbtp': a.true_peak, 'measured_lufs': float(after['input_i']), 'measured_dbtp': float(after['input_tp']), 'lra_lu': float(after['input_lra']), 'pcm_bits': 24, 'sample_rate': 48000, 'channels': 2}
        data['within_target'] = abs(data['measured_lufs']-a.lufs)<=0.5 and data['measured_dbtp']<=a.true_peak+0.1
        with dest.open('xb') as f, master.open('rb') as src:
            shutil.copyfileobj(src, f)
        with report.open('x') as f:
            json.dump(data, f, indent=2); f.write('\n')
    print(json.dumps(data))

if __name__ == '__main__':
    main()
