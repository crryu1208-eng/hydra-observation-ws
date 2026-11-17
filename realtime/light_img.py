"""light_img.py (host python with numpy / opencv) <frames.jsonl> <out dir> SYM=x0,y0,z0,x1,y1,z1 ...: ceiling light boxes from the images: in the frames
that see the light's surroundings (box + 0.8 m) best, white pixels (gray >= 235, low saturation) in blobs >= 150 px,
put in 3D with the depth (the blob's own, else the ceiling around it); the blobs whose 3D centre lies within 0.8 m
(x / y) of the box; box of their points (2-98%) in the frame with the most such pixels. Draws that frame."""
import json,os,sys,cv2,numpy as np
FR,OUT=sys.argv[1:3]; os.makedirs(OUT,exist_ok=True)
base=os.path.dirname(os.path.abspath(FR)); cal=json.load(open(f'{base}/camera_calib.json'))['sensors']['camera']
def R(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
F=[]
for k,l in enumerate(open(FR)):
    d=json.loads(l)
    c_=d.get('corrected') or {}
    if c_.get('status')=='ok': d['world_T_sensor']=c_['world_T_sensor']
    if d.get('status')=='written' and k%2==0: F.append((d['timestamp_ns'],f"{base}/{d['rgb_path']}",f"{base}/{d['depth_path']}",R(d['world_T_sensor']['q_wxyz']),np.array(d['world_T_sensor']['t'])))
fx,fy,cx,cy=cal['fx'],cal['fy'],cal['cx'],cal['cy']
res={}
for arg in sys.argv[3:]:
    sy,v=arg.split('='); b=np.array([float(x) for x in v.split(',')]); lo,hi=b[:3],b[3:]
    elo,ehi=lo-[0.8,0.8,0.3],hi+[0.8,0.8,0.3]
    C=np.array([[x,y,z] for x in (elo[0],ehi[0]) for y in (elo[1],ehi[1]) for z in (elo[2],ehi[2])]); c=(lo+hi)/2
    cand=[]
    for st,rp,dp,Rm,t in F:
        p=Rm.T@(c-t)
        if p[2]<0.5 or p[2]>6: continue
        P=(C-t)@Rm; P[:,2]=np.maximum(P[:,2],0.1)
        u=fx*P[:,0]/P[:,2]+cx; v=fy*P[:,1]/P[:,2]+cy
        x0,x1,y0,y1=[int(np.clip(q,0,W)) for q,W in ((u.min(),848),(u.max(),848),(v.min(),480),(v.max(),480))]
        if (x1-x0)*(y1-y0)<2000: continue
        cand.append(((x1-x0)*(y1-y0),st,rp,dp,Rm,t,(x0,y0,x1,y1)))
    cand.sort(key=lambda q:-q[0]); best=None
    for _,st,rp,dp,Rm,t,(x0,y0,x1,y1) in cand[:40]:
        im=cv2.imread(rp); z=cv2.imread(dp,-1).astype(np.float32)/1000
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); sat=im.max(2).astype(int)-im.min(2)
        m=np.zeros(g.shape,np.uint8); roi=(slice(y0,y1),slice(x0,x1))
        m[roi]=((g[roi]>=235)&(sat[roi]<=40)).astype(np.uint8)
        n,lab,stats,_=cv2.connectedComponentsWithStats(m)
        pts=[];pix=0;blobs=[]
        for i in range(1,n):
            if stats[i,4]<150: continue
            vv,uu=np.nonzero(lab==i); zz=z[vv,uu]; ok=(zz>0.3)&(zz<8)
            if ok.mean()<0.3:  # no depth on the lamp: the ceiling around it
                ring=cv2.dilate((lab==i).astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)&~(lab==i)
                zr=z[ring]; zr=zr[(zr>0.3)&(zr<8)]
                if len(zr)<20: continue
                zz=np.full(len(vv),np.median(zr)); ok=np.ones(len(vv),bool)
            Q=np.stack([(uu[ok]-cx)*zz[ok]/fx,(vv[ok]-cy)*zz[ok]/fy,zz[ok]],1)@Rm.T+t
            mc=np.median(Q,0)
            if np.all(mc[:2]>=lo[:2]-0.8) and np.all(mc[:2]<=hi[:2]+0.8) and mc[2]>1.8:
                pts.append(Q); pix+=len(Q); blobs.append(i)
        if pts and (best is None or pix>best[0]):
            best=(pix,st,rp,np.concatenate(pts),lab,blobs,Rm,t)
    if best is None: print(sy,'no white blob'); continue
    pix,st,rp,Q,lab,blobs,_,_=best
    zm=np.median(Q[:,2])  # a flat panel on the ceiling: every white pixel's ray cut with the plane z = median height
    Rm,t=best[6],best[7]; vv,uu=np.nonzero(np.isin(lab,blobs))
    d=np.stack([(uu-cx)/fx,(vv-cy)/fy,np.ones(len(uu))],1)@Rm.T; s=(zm-t[2])/d[:,2]; ok=s>0
    Q=t+s[ok,None]*d[ok]; Q=Q[np.all((Q[:,:2]>=lo[:2]-0.8)&(Q[:,:2]<=hi[:2]+0.8),1)]
    blo,bhi=np.percentile(Q,2,0),np.percentile(Q,98,0); blo[2]=min(blo[2],zm-0.05); bhi[2]=max(bhi[2],zm+0.05); res[sy]=list(blo)+list(bhi)
    im=cv2.imread(rp); mask=np.isin(lab,blobs); im[mask]=(0.5*im[mask]+[0,0,127]).astype(np.uint8)
    cv2.putText(im,f'{sy} {st}',(5,20),0,0.6,(0,255,255),2); cv2.imwrite(f'{OUT}/{sy}.jpg',im)
    print(sy,st,'pix',pix,np.round(blo,2),np.round(bhi,2),'size',np.round(bhi-blo,2))
print(';'.join(f"{k}="+','.join('%.3f'%x for x in v) for k,v in res.items()))
