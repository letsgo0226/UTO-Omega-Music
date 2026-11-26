"""ZPF-inspired classical finite-band field experiment, not quantum vacuum simulation."""
import argparse, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import neural as N
import music


def modes(rng, low, high, count):
    f = np.linspace(low, high, count)
    return f, np.sqrt(2*f/f.sum()), rng.uniform(0, 2*np.pi, count)


def field(t, spec):
    f, a, p = spec
    y = np.zeros_like(t)
    for k, w, q in zip(f, a, p):
        y += w*np.cos(2*np.pi*k*t+q)
    return y


def sing(m, notes, rate, bpm, length, tokens, seed, e, vib, strength):
    if strength == 0:
        return N.sing(m, notes, rate, bpm, length, tokens, seed, e, vib)
    rng = np.random.default_rng(seed)
    z = np.random.default_rng(np.random.SeedSequence([seed, 257, 1]))
    slow = modes(z, .5, 30, 24)
    air = modes(z, 400, 4000, 48)
    out = np.zeros(length, np.float32)
    for j, (b,d,n,vel) in enumerate(notes):
        start = round(b/music.PPQ*60/bpm*rate)
        count = min(length-start, round(d/music.PPQ*60/bpm*rate))
        if count <= 0: continue
        t = np.arange(count)/rate; dur = count/rate
        pitch = n-12; f = 440*2**((pitch-69)/12)
        v = N.V.index(tokens[j%len(tokens)])
        with N.torch.no_grad():
            weights = m(N.features(np.array([pitch]), np.array([v]), np.array([e])))[0].numpy()
        cents = vib*np.clip((t-.1)/.2,0,1)*np.sin(2*np.pi*5.2*t)
        cents += 8*strength*np.tanh(field(t+start/rate, slow))
        phase = 2*np.pi*np.cumsum(f*2**(cents/1200))/rate
        y = np.zeros(count)
        for h,w in enumerate(weights,1):
            if h*f < rate*.45: y += w*np.sin(h*phase)
        env = np.minimum(t/.025,1)*np.clip((dur-t)/min(.07,dur/3),0,1)
        breath = .013*rng.normal(size=count)+.003*strength*field(t+start/rate, air)
        out[start:start+count] += ((y+breath)*env*(vel/127)*.6).astype(np.float32)
    return np.column_stack((out,out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='zpf-song'); p.add_argument('--style',choices=music.STYLES,default='cinematic')
    p.add_argument('--key',choices=music.KEYS,default='D'); p.add_argument('--bars',type=int,default=4)
    p.add_argument('--bpm',type=int,default=96); p.add_argument('--seed',type=int,default=777)
    p.add_argument('--steps',type=int,default=180); p.add_argument('--vowels',default='a e i o u')
    p.add_argument('--strength',type=float,default=.75); p.add_argument('--expression',type=float,default=.6)
    p.add_argument('--vibrato',type=float,default=25); p.add_argument('--stems',action='store_true')
    p.add_argument('--model'); p.add_argument('--master',action='store_true')
    a = p.parse_args(); tokens = a.vowels.split()
    if not tokens or any(x not in N.V or len(x)!=1 for x in tokens): p.error('Use separate a e i o u tokens')
    if not (1<=a.bars<=64 and 40<=a.bpm<=220 and a.bars*240/a.bpm<=180 and 0<=a.seed<2**32-1 and 1<=a.steps<=3000 and 0<=a.strength<=1 and 0<=a.expression<=1 and 0<=a.vibrato<=100): p.error('Parameter out of range')
    if a.master and not shutil.which('ffmpeg'): p.error('Install FFmpeg separately')
    N.torch.set_num_threads(1); N.torch.use_deterministic_algorithms(True)
    if a.model:
        m=N.model()
        with np.load(a.model,allow_pickle=False) as z: m.load_state_dict({k:N.torch.from_numpy(z[k].copy()) for k in m.state_dict()})
        m.eval(); metrics={'data':'local external weights; provenance not verified'}
    else: m,metrics=N.train(a.seed,a.steps)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    base=out/'baseline'; test=out/'field'; base.mkdir(); test.mkdir()
    with (out/'neural-weights.npz').open('xb') as f: np.savez(f,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
    tracks=music.compose(a.style,a.key,a.bars,a.seed)
    music.midi(base/'score.mid',tracks,a.bpm,a.style)
    audio=music.render(tracks,a.bpm,a.bars,48000,a.seed,base,a.stems,a.style)
    for pth in base.iterdir():
        if pth.is_file(): shutil.copyfile(pth,test/pth.name)
    rate,mix=N.read(base/'mix.wav')
    args=(m,tracks[0],rate,a.bpm,len(mix),tokens,a.seed,a.expression,a.vibrato)
    v0=sing(*args,0); v1=sing(*args,a.strength)
    b0=mix*.65+v0; b1=mix*.65+v1
    gain=min(1,10**(-1/20)/max(float(np.max(np.abs(b0))),float(np.max(np.abs(b1))),1e-12))
    report={'scope':'classical seeded modal field; not quantum energy extraction or quality evidence','seed':a.seed,'strength':a.strength,'settings':vars(a),'training':metrics,'audio':audio,'shared_gain':gain,'pitch_bound_cents':8*a.strength,'vocal_delta_rms':float(np.sqrt(np.mean((v1-v0)**2))),'baseline_vocal_rms':float(np.sqrt(np.mean(v0**2))),'masters':{}}
    for name,d,v,b in [('baseline',base,v0,b0),('field',test,v1,b1)]:
        N.write(d/'vocal.wav',v*gain,rate); N.write(d/'vocal-mix.wav',b*gain,rate)
        if a.master:
            subprocess.run([sys.executable,str(Path(__file__).with_name('studio.py')),str(d)],check=True)
            report['masters'][name]=json.loads((d/'studio-report.json').read_text())
    (out/'zpf-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__': main()
