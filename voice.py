"""Offline original formant singing and authorized PCM vocal mixing."""
import argparse, json, math, re, subprocess, sys, wave
from pathlib import Path
import numpy as np
import music

FORMANTS={'a':(800,1150,2900),'e':(450,1900,2700),'i':(300,2300,3100),'o':(500,800,2600),'u':(350,650,2400)}

def read_pcm(path):
    with wave.open(str(path),'rb') as w:
        rate,ch,bits=w.getframerate(),w.getnchannels(),w.getsampwidth()
        if ch not in (1,2) or bits not in (2,3):raise ValueError('PCM WAV: mono/stereo, 16/24 bit required')
        data=w.readframes(w.getnframes())
    if bits==2:a=np.frombuffer(data,'<i2').astype(np.float32)/32768
    else:
        b=np.frombuffer(data,np.uint8).reshape(-1,3).astype(np.int32)
        q=b[:,0]+(b[:,1]<<8)+(b[:,2]<<16);a=((q^8388608)-8388608).astype(np.float32)/8388608
    a=a.reshape(-1,ch)
    return rate,np.repeat(a,2,axis=1) if ch==1 else a

def write_pcm(path,rate,a):
    q=np.rint(np.clip(a,-1,1)*8388607).astype('<i4')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(2);w.setsampwidth(3);w.setframerate(rate)
        w.writeframes(q.view(np.uint8).reshape(-1,4)[:,:3].tobytes())

def syllables(s):
    tokens=s.lower().split()
    if not tokens or any(not re.fullmatch(r'[mnlrsvfhptkbdgwyz]*[aeiou][mn]?',x) for x in tokens):
        raise ValueError('Use space-separated simple vowel syllables, e.g. la lu mi no; not free-text lyrics')
    return tokens

def sing(notes,bpm,rate,length,tokens,seed,octave,vibrato):
    rng=np.random.default_rng(seed);out=np.zeros(length,np.float32)
    for j,(b,d,n,v) in enumerate(notes):
        start=round(b/music.PPQ*60/bpm*rate);dur=d/music.PPQ*60/bpm
        count=min(length-start,round(dur*rate))
        if count<=1:continue
        t=np.arange(count)/rate;word=tokens[j%len(tokens)];vowel=next(c for c in word if c in FORMANTS)
        f=440*2**((n+12*octave-69)/12);rise=np.clip((t-.11)/.15,0,1)
        phase=2*np.pi*np.cumsum(f*2**(vibrato*rise*np.sin(2*np.pi*5.3*t)/1200))/rate
        y=np.zeros(count);centers=FORMANTS[vowel]
        for h in range(1,min(65,int(rate*.45/f))+1):
            freq=h*f;weight=.09+sum(g*math.exp(-.5*((freq-c)/bw)**2) for c,bw,g in zip(centers,(110,160,240),(1,.7,.35)))
            y+=weight/h**.85*np.sin(h*phase)
        y/=max(1,float(np.max(np.abs(y))))
        onset=min(.055,dur*.22);noise=rng.normal(0,1,count)
        consonant=np.exp(-t/max(.008,onset/2)) if word[0] not in FORMANTS else 0
        env=np.minimum(t/.025,1)*np.clip((dur-t)/min(.065,dur*.4),0,1)
        y=(y+.025*noise+.11*consonant*noise)*env*(v/127)*.35
        out[start:start+count]+=y.astype(np.float32)
    return np.column_stack((out,out))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voice',choices=['synth','wav'],required=True)
    p.add_argument('--syllables',default='la lo mi na lu e ma o')
    p.add_argument('--voice-wav');p.add_argument('--authorized',action='store_true')
    p.add_argument('--voice-gain',type=float,default=1.0)
    p.add_argument('--voice-offset',type=float,default=0.0)
    p.add_argument('--voice-octave',type=int,choices=[-2,-1,0,1],default=0)
    p.add_argument('--vibrato',type=float,default=25)
    a,rest=p.parse_known_args()
    if '--midi-only' in rest:p.error('vocal mode requires audio rendering')
    if not (0<=a.voice_gain<=4 and 0<=a.voice_offset<=300 and 0<=a.vibrato<=100):p.error('gain 0..4, offset 0..300, vibrato 0..100 cents')
    try:tokens=syllables(a.syllables) if a.voice=='synth' else []
    except ValueError as e:p.error(str(e))
    if a.voice=='wav' and not (a.voice_wav and a.authorized):p.error('provide --voice-wav and --authorized for audio you may use')
    if a.voice=='synth' and a.voice_wav:p.error('--voice-wav requires --voice wav')
    base=argparse.ArgumentParser(add_help=False);base.add_argument('--out',default='music-output');b,_=base.parse_known_args(rest)
    out=Path(b.out);ext=read_pcm(a.voice_wav) if a.voice=='wav' else None
    subprocess.run([sys.executable,str(Path(__file__).with_name('music.py')),*rest],check=True)
    meta=json.loads((out/'session.json').read_text());rate,mix=read_pcm(out/'mix.wav')
    if a.voice=='synth':
        notes=music.compose(meta['style'],meta['key'],meta['bars'],meta['seed'])[0]
        raw=sing(notes,meta['bpm'],rate,len(mix),tokens,meta['seed'],a.voice_octave,a.vibrato)
    else:
        if ext[0]!=rate:raise ValueError('Vocal WAV rate must match --rate; resample in your DAW first')
        raw=ext[1]
    voice=np.zeros_like(mix);offset=round(a.voice_offset*rate);n=min(len(raw),max(0,len(mix)-offset))
    if n<=0:raise ValueError('Vocal starts outside the backing track')
    voice[offset:offset+n]=raw[:n]*a.voice_gain
    combined=mix*.7+voice;peak=float(np.max(np.abs(combined)))
    gain=min(1,10**(-1/20)/max(peak,1e-12))
    write_pcm(out/'vocal.wav',rate,voice*gain);write_pcm(out/'vocal-mix.wav',rate,combined*gain)
    meta['voice']={'type':a.voice,'syllables':tokens,'gain':a.voice_gain,'offset_seconds':a.voice_offset,'octave':a.voice_octave,'vibrato_cents':a.vibrato,'mix_gain':gain,'peak_dbfs':20*math.log10(max(peak*gain,1e-12)),'sample_rate':rate,'frames':len(mix),'backing':'mix.wav','mixed':'vocal-mix.wav','stem':'vocal.wav'}
    (out/'session.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['voice']))
if __name__=='__main__':main()
