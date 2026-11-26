"""UTO-Omega-Music: deterministic original procedural composition, MIDI and PCM."""
import argparse, json, math, random, struct, wave
from pathlib import Path
import numpy as np

PPQ=480
KEYS={'C':0,'C#':1,'D':2,'Eb':3,'E':4,'F':5,'F#':6,'G':7,'Ab':8,'A':9,'Bb':10,'B':11}
STYLES={
 'ambient': (72,[0,5,2,4],[0,2,3,5,7,8,10],(88,89,38,10)),
 'cinematic': (96,[0,5,3,4],[0,2,3,5,7,8,10],(48,46,43,9)),
 'electronic': (120,[0,5,3,4],[0,2,3,5,7,8,10],(81,89,38,12)),
 'jazz': (112,[1,4,0,5],[0,2,4,5,7,9,11],(0,4,32,11)),
}
NAMES=['melody','harmony','bass','counterpoint','drums']

def compose(style,key,bars,seed):
    rng=random.Random(seed); _,progression,scale,_=STYLES[style]; root=KEYS[key]; tracks=[[] for _ in NAMES]
    def add(ch,b,d,n,v):
        if b+d<=bars*4+1e-8: tracks[ch].append((round(b*PPQ),max(1,round(d*PPQ)),int(n),int(v)))
    def note(degree,octave): return 12*octave+root+scale[degree%7]+12*(degree//7)
    motif=[rng.choice([0,2,4]),rng.choice([1,2,3]),rng.choice([2,4,5]),rng.choice([0,2,4])]
    for bar in range(bars):
        b=bar*4; d=progression[bar%4]; ending=bar==bars-1
        if ending:d=0
        section=min(3,4*bar//bars); intensity=[.70,.88,1,.72][section]
        chord=[d,d+2,d+4]+([d+6] if style=='jazz' else [])
        # Smooth, voiced chord bed. Final cadence resolves to tonic.
        for j,k in enumerate(chord): add(1,b+(j*.06 if style=='jazz' else 0),3.65-j*.06,note(k,4),int(58*intensity))
        for off in ([0,2] if style=='ambient' else [0,1.5,2,3.5]):
            add(2,b+off,.85 if style!='ambient' else 1.7,note(d if off<2 else d+4,2),int(82*intensity))
        offsets=[0,1,2,3] if style=='ambient' else [0,.75,1.5,2.5,3.25]
        if ending: offsets=[0,1,2]
        for j,off in enumerate(offsets):
            deg=(motif[j%4]+d if bar%4 in (0,2) else rng.choice(chord+[d+1,d+3]))
            if ending and j==len(offsets)-1:deg=0
            if section==2 and not ending:deg+=7 if j==2 else 0
            dur=min((offsets[j+1]-off)*.91 if j+1<len(offsets) else 3.9-off,1.8)
            add(0,b+off,dur,note(deg,5),int((76+rng.randrange(18))*intensity))
        if section in (1,2) or style=='jazz':
            for j in range(4):add(3,b+j+.25,.52,note(chord[j%len(chord)],5),int(43*intensity))
        if style!='ambient':
            for off in ([0,2] if style!='electronic' else [0,1,2,3]):add(4,b+off,.18,36,int(90*intensity))
            for off in [1,3]:add(4,b+off,.14,38,int(72*intensity))
            for j in range(8):add(4,b+j/2,.07,42,int((42+8*(j%2))*intensity))
    return tracks

def vlq(n):
    a=[n&127]; n>>=7
    while n:a.append(128|(n&127));n>>=7
    return bytes(reversed(a))
def chunk(tag,b):return tag+struct.pack('>I',len(b))+b
def midi(path,tracks,bpm,style):
    tempo=round(60000000/bpm); body=[chunk(b'MTrk',b'\x00\xff\x51\x03'+tempo.to_bytes(3,'big')+b'\x00\xff\x58\x04\x04\x02\x18\x08\x00\xff\x2f\x00')]
    for ch,notes in enumerate(tracks):
        c=9 if ch==4 else ch; name=NAMES[ch].encode(); out=bytearray(b'\x00\xff\x03'+vlq(len(name))+name)
        if ch!=4:out.extend(bytes([0,192+c,STYLES[style][3][ch]]))
        events=[]
        for b,d,n,v in notes:events.extend([(b,1,n,v),(b+d,0,n,0)])
        last=0
        for b,on,n,v in sorted(events):out.extend(vlq(b-last)+bytes([(144 if on else 128)+c,n,v]));last=b
        out.extend(b'\x00\xff\x2f\x00');body.append(chunk(b'MTrk',out))
    path.write_bytes(chunk(b'MThd',struct.pack('>HHH',1,len(body),PPQ))+b''.join(body))

def render(tracks,bpm,bars,rate,seed,out,stems,style):
    beat=60/bpm; length=math.ceil((bars*4*beat+1.2)*rate); mix=np.zeros((length,2),np.float32); rng=np.random.default_rng(seed)
    pans=[-.16,.24,0,-.48,.08]; gains=[.24,.14,.22,.12,.22]; saved=[]
    for ch,notes in enumerate(tracks):
        if not notes:continue
        s=np.zeros_like(mix); pan=pans[ch]; left=math.sqrt((1-pan)/2);right=math.sqrt((1+pan)/2)
        for b,d,n,v in notes:
            start=round(b/PPQ*beat*rate); dur=d/PPQ*beat; release=.24 if ch==1 else .09
            if ch==4:release=.03
            t=np.arange(round((dur+release)*rate),dtype=np.float64)/rate; f=440*2**((n-69)/12); phase=2*np.pi*f*t
            attack=.065 if ch==1 else .012
            env=np.minimum(t/attack,1)*np.clip((dur+release-t)/release,0,1)
            if ch==0: y=np.sin(phase+.14*np.sin(phase*2))+.20*np.sin(2*phase)+.07*np.sin(3*phase)
            elif ch==1:y=(np.sin(phase)+.38*np.sin(phase*1.003)+.28*np.sin(2*phase))/1.66
            elif ch==2:y=np.sin(phase)+.18*np.sin(2*phase)
            elif ch==3:y=(np.sin(phase)+.24*np.sin(3*phase))*np.exp(-3*t)
            elif n==36:y=np.sin(2*np.pi*(48*t+4*(1-np.exp(-25*t))))*np.exp(-18*t)
            elif n==38:y=(.75*rng.uniform(-1,1,len(t))+.25*np.sin(2*np.pi*180*t))*np.exp(-19*t)
            else:y=rng.uniform(-1,1,len(t))*np.exp(-48*t)
            y=(y*env*(v/127)*gains[ch]).astype(np.float32); end=min(length,start+len(y));y=y[:end-start];s[start:end,0]+=y*left;s[start:end,1]+=y*right
        if ch in (0,1,3):
            delay=round(rate*.23);s[delay:]+=s[:-delay,::-1].copy()*.12
        mix+=s
        if stems:
            # Store float intermediates on disk so RAM does not scale with stems.
            temp=out/(NAMES[ch]+'.npy');np.save(temp,s);saved.append((temp,NAMES[ch]))
    peak=float(np.max(np.abs(mix))); gain=min(4,10**(-1/20)/max(peak,1e-12))
    def write(path,a):
        with wave.open(str(path),'wb') as w:w.setnchannels(2);w.setsampwidth(3);w.setframerate(rate);q=np.rint(np.clip(a*gain,-1,1)*8388607).astype('<i4');w.writeframes(q.view(np.uint8).reshape(-1,4)[:,:3].tobytes())
    write(out/'mix.wav',mix)
    for p,name in saved:write(out/(name+'.wav'),np.load(p));p.unlink()
    return {'sample_rate':rate,'channels':2,'pcm_bits':24,'seconds':length/rate,'peak_dbfs':20*math.log10(max(peak*gain,1e-12)),'mix_gain':gain}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--style',choices=STYLES,default='ambient');p.add_argument('--key',choices=KEYS,default='D');p.add_argument('--bars',type=int,default=16);p.add_argument('--bpm',type=int);p.add_argument('--seed',type=int,default=7);p.add_argument('--rate',type=int,choices=[44100,48000],default=48000);p.add_argument('--out',default='music-output');p.add_argument('--stems',action='store_true');p.add_argument('--midi-only',action='store_true');a=p.parse_args()
    bpm=STYLES[a.style][0] if a.bpm is None else a.bpm
    if not 1<=a.bars<=128 or not 40<=bpm<=220 or not 0<=a.seed<2**32:p.error('bars 1..128, bpm 40..220, seed 0..2^32-1')
    if not a.midi_only and a.bars*240/bpm>300:p.error('audio limit 300 seconds; use --midi-only for longer pieces')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False); tracks=compose(a.style,a.key,a.bars,a.seed);midi(out/'score.mid',tracks,bpm,a.style)
    meta={'engine':'UTO-Omega-Music/0.1','style':a.style,'key':a.key,'bars':a.bars,'bpm':bpm,'seed':a.seed,'notes':dict(zip(NAMES,map(len,tracks))), 'audio':None}
    if not a.midi_only:meta['audio']=render(tracks,bpm,a.bars,a.rate,a.seed,out,a.stems,a.style)
    (out/'session.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta));print(out.resolve())
if __name__=='__main__':main()
