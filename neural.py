"""Neural harmonic singing prototype; synthetic training only."""
import argparse, json, wave
from pathlib import Path
import numpy as np
import torch
from torch import nn
import music

V='aeiou'
H=32
F=np.array([[800,1150,2900],[450,1900,2700],[300,2300,3100],[500,800,2600],[350,650,2400]])
def features(p,v,e):
    return torch.tensor(np.column_stack(((p-60)/24,np.eye(5)[v],e)),dtype=torch.float32)
def data(seed,n):
    r=np.random.default_rng(seed);p=r.uniform(40,90,n);v=r.integers(0,5,n);e=r.random(n);h=np.arange(1,H+1)
    f=440*2**((p-69)/12);z=f[:,None]*h
    g=np.exp(-.5*((z[:,:,None]-F[v,None,:]*(1+.08*e[:,None,None]))/np.array([120,180,260]))**2)
    y=(.08+g@np.array([1,.7,.35]))/h**(1.2-.45*e[:,None]);y/=y.sum(axis=1,keepdims=True)
    return features(p,v,e),torch.tensor(y,dtype=torch.float32)
def model():
    return nn.Sequential(nn.Linear(7,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,H),nn.Softmax(dim=-1))
def train(seed,steps):
    torch.manual_seed(seed);m=model();x,y=data(seed,2048);u,v=data(seed+1,512);loss=nn.MSELoss();opt=torch.optim.Adam(m.parameters(),lr=.008)
    with torch.no_grad():before=loss(m(u),v).item()
    for _ in range(steps):
        i=torch.randint(len(x),(128,));l=loss(m(x[i]),y[i]);opt.zero_grad();l.backward();opt.step()
    with torch.no_grad():after=loss(m(u),v).item()
    return m.eval(),{'before_mse':before,'after_mse':after,'steps':steps,'train':len(x),'validation':len(u),'parameters':sum(p.numel() for p in m.parameters())}
def read(path):
    with wave.open(str(path),'rb') as w:
        assert w.getsampwidth()==3 and w.getnchannels()==2
        rate=w.getframerate();b=np.frombuffer(w.readframes(w.getnframes()),np.uint8).reshape(-1,3).astype(np.int32)
    q=b[:,0]+(b[:,1]<<8)+(b[:,2]<<16)
    return rate,(((q^8388608)-8388608)/8388608).astype(np.float32).reshape(-1,2)
def write(path,a,rate):
    q=np.rint(np.clip(a,-1,1)*8388607).astype('<i4')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(2);w.setsampwidth(3);w.setframerate(rate);w.writeframes(q.view(np.uint8).reshape(-1,4)[:,:3].tobytes())
def sing(m,notes,rate,bpm,length,tokens,seed,e,vib):
    rng=np.random.default_rng(seed);out=np.zeros(length,np.float32)
    for j,(b,d,n,vel) in enumerate(notes):
        start=round(b/music.PPQ*60/bpm*rate);count=min(length-start,round(d/music.PPQ*60/bpm*rate))
        if count<=0:continue
        t=np.arange(count)/rate;dur=count/rate;pitch=n-12;f=440*2**((pitch-69)/12);v=V.index(tokens[j%len(tokens)])
        with torch.no_grad():weights=m(features(np.array([pitch]),np.array([v]),np.array([e])))[0].numpy()
        phase=2*np.pi*np.cumsum(f*2**(vib*np.clip((t-.1)/.2,0,1)*np.sin(2*np.pi*5.2*t)/1200))/rate
        y=np.zeros(count)
        for h,w in enumerate(weights,1):
            if h*f<rate*.45:y+=w*np.sin(h*phase)
        env=np.minimum(t/.025,1)*np.clip((dur-t)/min(.07,dur/3),0,1)
        out[start:start+count]+=((y+.013*rng.normal(size=count))*env*(vel/127)*.6).astype(np.float32)
    return np.column_stack((out,out))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='neural-song');p.add_argument('--style',choices=music.STYLES,default='cinematic');p.add_argument('--key',choices=music.KEYS,default='D');p.add_argument('--bpm',type=int,default=96);p.add_argument('--bars',type=int,default=4);p.add_argument('--seed',type=int,default=777);p.add_argument('--steps',type=int,default=180);p.add_argument('--vowels',default='a e i o u');p.add_argument('--expression',type=float,default=.6);p.add_argument('--vibrato',type=float,default=25);p.add_argument('--model');p.add_argument('--stems',action='store_true');a=p.parse_args();tokens=a.vowels.split()
    if not tokens or any(x not in list(V) for x in tokens):p.error('space-separated a e i o u only')
    if not (1<=a.bars<=64 and 40<=a.bpm<=220 and a.bars*240/a.bpm<=180 and 0<=a.seed<2**32-1 and 1<=a.steps<=3000 and 0<=a.expression<=1 and 0<=a.vibrato<=100):p.error('parameter out of range')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.model:
        m=model()
        with np.load(a.model,allow_pickle=False) as z:m.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in m.state_dict()})
        m.eval();metrics={'data':'external local weights; provenance not verified'}
    else:m,metrics=train(a.seed,a.steps)
    with (out/'neural-weights.npz').open('xb') as f:np.savez(f,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
    tracks=music.compose(a.style,a.key,a.bars,a.seed);music.midi(out/'score.mid',tracks,a.bpm,a.style);audio=music.render(tracks,a.bpm,a.bars,48000,a.seed,out,a.stems,a.style)
    rate,mix=read(out/'mix.wav');voice=sing(m,tracks[0],rate,a.bpm,len(mix),tokens,a.seed,a.expression,a.vibrato);mix=mix*.65+voice;gain=min(1,10**(-1/20)/max(float(np.max(np.abs(mix))),1e-12))
    write(out/'vocal.wav',voice*gain,rate);write(out/'vocal-mix.wav',mix*gain,rate)
    report={'model':'UTO-NeuralVoice/0.1','training':metrics,'audio':audio,'vowels':tokens,'seed':a.seed,'expression':a.expression,'vibrato_cents':a.vibrato,'mix_gain':gain,'scope':'synthetic neural harmonic envelope plus DSP, not human-quality singing'}
    (out/'neural-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
