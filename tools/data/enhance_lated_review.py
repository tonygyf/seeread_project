"""Build native-resolution, non-generative human review views for the frozen pilot."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from seeread.v5.lated_annotations import read_jsonl, render_pages, verify_package, write_jsonl
from seeread.v5.reading import sha256


def balance_bright(bright):
    gray=cv2.cvtColor(bright,cv2.COLOR_BGR2GRAY)
    low,high=np.percentile(gray,(1,99.5))
    pixels=np.clip((bright.astype(np.float32)-low)/max(float(high-low),3.),0,1)
    return np.round(pixels**.85*255).astype(np.uint8)


def human_views(original, bright):
    """Every output derives from supplied pixels; no OCR or character synthesis."""
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gain = min(64., 210. / max(float(np.percentile(gray, 99)), 1.))
    exposure = np.clip(original.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    exposure = cv2.LUT(exposure, np.round((np.arange(256) / 255.) ** .8 * 255).astype(np.uint8))
    lab = cv2.cvtColor(exposure, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    local = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    smooth = cv2.bilateralFilter(bright, 5, 18, 3)
    detail = cv2.addWeighted(smooth, 1.6, cv2.GaussianBlur(smooth, (0, 0), .9), -.6, 0)
    intensity = gray.astype(np.float32) + 1
    factor=min(1.,768/max(gray.shape))
    small=cv2.resize(intensity,(max(1,round(gray.shape[1]*factor)),max(1,round(gray.shape[0]*factor))))
    illumination=sum(cv2.resize(np.log(cv2.GaussianBlur(small,(0,0),s)),(gray.shape[1],gray.shape[0])) for s in (15.,80.))/2
    retinex = np.log(intensity) - illumination
    low, high = np.percentile(retinex, (1, 99))
    retinex = np.clip((retinex - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
    return dict(balanced=balance_bright(bright),bright=bright, exposure=exposure, local=local, detail=detail,
                retinex=cv2.cvtColor(retinex, cv2.COLOR_GRAY2BGR))


def expanded_reference(bright, bbox):
    x0,y0,x1,y1=bbox
    margin=max(32,round(.15*min(x1-x0,y1-y0)))
    frame=[max(0,x0-margin),max(0,y0-margin),min(bright.shape[1],x1+margin),min(bright.shape[0],y1+margin)]
    a,b,c,d=frame
    return bright[b:d,a:c],frame


def viewing_score(image, mask, reference_edges=None):
    """Rank visible, coherent stroke contrast while penalizing fine noise."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    smooth=cv2.GaussianBlur(gray,(0,0),.8)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=.125)
    xx=cv2.GaussianBlur(gx*gx,(0,0),1.5);yy=cv2.GaussianBlur(gy*gy,(0,0),1.5)
    xy=cv2.GaussianBlur(gx*gy,(0,0),1.5)
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)
    edge=np.sqrt(gx*gx+gy*gy)
    inside=mask>0
    values=smooth[inside]
    if not values.size:
        raise ValueError('Empty target polygon in viewing score')
    low,high=np.percentile(values,(5,95))
    quiet=inside & (edge<=np.percentile(edge[inside],50))
    noise=float(np.median(np.abs(gray-smooth)[quiet]))
    strokes=float(np.percentile((edge*np.sqrt(coherence))[inside],90))
    score=float(high-low+3*strokes-2*noise)
    if reference_edges is not None:
        rx,ry=reference_edges
        dot=float(np.sum((gx*rx+gy*ry)[inside]))
        norm=float(np.sqrt(np.sum((gx*gx+gy*gy)[inside])*np.sum((rx*rx+ry*ry)[inside])))
        fidelity=abs(dot)/max(norm,1e-8)
        score*=min(1.,fidelity)**2
    return score


def tone_pixels(image, ev=0., gamma=1., contrast=1., channel='rgb'):
    if channel!='rgb':
        index={'red':2,'green':1,'blue':0}[channel]
        image=cv2.cvtColor(image[:,:,index],cv2.COLOR_GRAY2BGR)
    lut=np.clip(((np.minimum(1,np.arange(256)*2**ev/255)**gamma)*255-127.5)*contrast+127.5,0,255)
    return cv2.LUT(image,np.round(lut).astype(np.uint8))


def detail_pixels(image, denoise=0., sharp=0.):
    if denoise:
        image=cv2.addWeighted(image,1-denoise,cv2.GaussianBlur(image,(0,0),.8),denoise,0)
    if sharp:
        image=cv2.addWeighted(image,1+sharp,cv2.GaussianBlur(image,(0,0),.8),-sharp,0)
    return image


def viewing_angle(poly):
    points=np.asarray(poly,np.float32)
    _,(w,h),angle=cv2.minAreaRect(points)
    if min(w,h)<=0 or max(w,h)/min(w,h)<2.5:
        return 0.
    if abs(cv2.contourArea(points))/(w*h)<.6:
        return 0.
    if w<h:angle+=90
    angle=(angle+90)%180-90
    if abs(angle)>45:
        return 0.  # Stacked upright glyphs cannot be distinguished from sideways words by a polygon.
    return round(-angle,1) if abs(angle)>2 else 0.


def choose_view(candidates, poly):
    h,w=next(iter(candidates.values())).shape[:2]
    factor=min(1.,max(640/max(h,w),24/min(h,w)))
    wh=(max(1,round(w*factor)),max(1,round(h*factor)))
    mask=np.zeros((wh[1],wh[0]),np.uint8)
    cv2.fillPoly(mask,[np.rint(np.asarray(poly)*factor).astype(np.int32)],255)
    small={name:cv2.resize(im,wh,interpolation=cv2.INTER_AREA) for name,im in candidates.items()}
    anchor=small.get('balanced',next(iter(small.values())))
    reference=cv2.GaussianBlur(cv2.cvtColor(anchor,cv2.COLOR_BGR2GRAY).astype(np.float32)/255,(0,0),.8)
    reference_edges=(cv2.Sobel(reference,cv2.CV_32F,1,0,ksize=3,scale=.125),cv2.Sobel(reference,cv2.CV_32F,0,1,ksize=3,scale=.125))
    score_view=lambda pixels:viewing_score(pixels,mask,reference_edges)
    ranked=sorted(small,key=lambda name:score_view(small[name]),reverse=True)
    presets=[(0,1,1),(0,.8,1),(0,1.2,1),(-.3,1,1.1),(.3,1,1),(0,1,1.2)]
    best=None
    for name in ranked[:2]:
        for channel in ('rgb','red','green','blue'):
            for ev,gamma,contrast in presets:
                params=dict(ev=ev,gamma=gamma,contrast=contrast,channel=channel)
                pixels=tone_pixels(small[name],**params)
                score=score_view(pixels)-(.025 if channel!='rgb' else 0)-.005*(abs(ev)+abs(gamma-1)+abs(contrast-1))
                if best is None or score>best['score']:
                    best=dict(source=name,score=score,parameters=params)
    toned=tone_pixels(small[best['source']],**best['parameters'])
    choice=(score_view(toned),0.,0.)
    for denoise in (0.,.25,.5):
        for sharp in (0.,.3,.6):
            score=score_view(detail_pixels(toned,denoise,sharp))-.01*(denoise+sharp)
            if score>choice[0]:choice=(score,denoise,sharp)
    best['parameters'].update(denoise=choice[1],sharp=choice[2])
    best['angle']=viewing_angle(poly)
    best['score']=choice[0]
    image=tone_pixels(candidates[best['source']],**{k:v for k,v in best['parameters'].items() if k not in ('denoise','sharp')})
    return detail_pixels(image,choice[1],choice[2]),best


def recommend(package):
    import time
    cv2.setNumThreads(2)
    package=Path(package);binding=sha256(package/'protocol.json');_,tasks=verify_package(package)
    task_by_id={t['id']:t for t in tasks if t['pilot']}
    records=read_jsonl(package/'human_views.jsonl')
    if {r['id'] for r in records}!=set(task_by_id):raise ValueError('Expected the complete frozen pilot')
    overrides_path=package/'human_view_overrides.json'
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    if set(overrides)-set(task_by_id):raise ValueError('Viewing override contains an unknown pilot instance')
    output=package/'human_auto_selected';output.mkdir(exist_ok=False)
    started=time.monotonic()
    for index,row in enumerate(records):
        task=task_by_id[row['id']]
        paths={k:v for k,v in row['views'].items() if k not in ('expanded','auto')}
        paths['original']=task['crop']
        candidates={name:cv2.imread(str(package/path),cv2.IMREAD_COLOR) for name,path in paths.items()}
        if any(im is None for im in candidates.values()):raise ValueError('Missing human viewing source')
        image,choice=choose_view(candidates,row['poly_local'])
        if row['id'] in overrides:
            choice['angle']=float(overrides[row['id']]['angle'])
            choice['visual_review']=overrides[row['id']]['reason']
        target=output/Path(task['crop']).name
        if not cv2.imwrite(str(target),image):raise OSError(target)
        row['views']['auto']=str(target.relative_to(package))
        row['recommendation']=dict(choice,revision='auto2',selection='heuristic human-view recommendation; not an OCR or legibility score',parameters_baked=True)
        if (index+1)%20==0:print(json.dumps(dict(instances=index+1,seconds=round(time.monotonic()-started,1))),flush=True)
    write_jsonl(package/'human_views.jsonl.tmp',records);(package/'human_views.jsonl.tmp').replace(package/'human_views.jsonl')
    path=package/'human_views_provenance.json';report=json.loads(path.read_text())
    report['automatic']=dict(status='complete',revision='auto2',instances=len(records),source_sha256=sha256(Path(__file__)),
                             ranking='GT-polygon-masked coherent gradient contrast minus fine-noise penalty, multiplied by squared edge agreement with supplied bright anchor; two source finalists,48 tone/channel candidates,9 detail candidates',
                             angle='deskew elongated near-rectangular lines within45 degrees; vertical/stacked/180 orientation not inferred',
                             parameters_baked=True,elapsed_seconds=time.monotonic()-started,no_ocr_or_transcriptions_used=True,
                             visual_overrides=overrides)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if binding!=sha256(package/'protocol.json'):raise ValueError('Frozen package identity changed')
    render_pages(package)
    return dict(status='complete',instances=len(records))


def prepare(package):
    cv2.setNumThreads(2)
    package = Path(package)
    protocol, tasks = verify_package(package)
    binding = sha256(package/'protocol.json')
    root = Path(protocol['data_root'])
    sources = {r['image']: r for source in protocol['sources'].values() for r in read_jsonl(source['path'])}
    selected = [t for t in tasks if t['pilot']]
    output = package/'human_views'
    output.mkdir(exist_ok=False)
    groups = {}
    for task in selected:
        groups.setdefault(task['image'], []).append(task)
    records, fingerprints = [], {}
    for image_path, items in groups.items():
        row = sources[image_path]
        original = cv2.imread(str(root/image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        bright = cv2.imread(str(root/row['lit']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if original is None or bright is None or original.shape != bright.shape:
            raise ValueError(f'Missing or differently sized supplied bright reference: {image_path}')
        fingerprints[row['lit']] = sha256(root/row['lit'])
        h, w = original.shape[:2]
        factor = min(1., 1600/max(h,w))
        context = cv2.resize(bright, (round(w*factor), round(h*factor)))
        for task in items:
            poly = np.asarray(task['poly'], np.float32)
            x0, y0 = np.maximum(np.floor(poly.min(0)-8).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(poly.max(0)+8).astype(int), [w,h])
            crop = original[y0:y1,x0:x1]
            if not np.array_equal(crop, cv2.imread(str(package/task['crop']))):
                raise ValueError(f'Human preview differs from frozen native crop: {task["id"]}')
            views = {}
            for name, pixels in human_views(crop, bright[y0:y1,x0:x1]).items():
                target = output/f'{Path(task["crop"]).stem}_{name}.png'
                if not cv2.imwrite(str(target), pixels):
                    raise OSError(target)
                views[name] = str(target.relative_to(package))
            expanded,frame=expanded_reference(bright,[x0,y0,x1,y1])
            expanded_path=output/f'{Path(task["crop"]).stem}_expanded.png'
            if not cv2.imwrite(str(expanded_path),expanded):
                raise OSError(expanded_path)
            views['expanded']=str(expanded_path.relative_to(package))
            context_path = output/f'{Path(task["context"]).stem}_context.jpg'
            records.append(dict(id=task['id'], views=views, context=str(context_path.relative_to(package)),
                                poly_local=(poly-[x0,y0]).tolist(), bbox=[int(x) for x in (x0,y0,x1,y1)],
                                view_bbox={'expanded':[int(x) for x in frame]},
                                bright_source=row['lit'], human_aid_only=True))
            points=np.rint(poly*factor).astype(np.int32)
            cv2.polylines(context,[points],True,(255,255,0),2)
            anchor=tuple(points[0])
            cv2.putText(context,str(task['instance_index']),anchor,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4)
            cv2.putText(context,str(task['instance_index']),anchor,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),1)
        if not cv2.imwrite(str(context_path),context,[cv2.IMWRITE_JPEG_QUALITY,95]):
            raise OSError(context_path)
        print(json.dumps(dict(images=len(fingerprints),instances=len(records))),flush=True)
    write_jsonl(package/'human_views.jsonl', records)
    report=dict(status='complete',package_sha256=binding,images=len(groups),instances=len(records),
                methods=dict(bright='dataset-supplied image_bright; exposure/enhancement origin not independently verified',
                             balanced='supplied bright, per-crop gray p1/p99.5 range stretch, gamma0.85; default human display',
                             exposure='original pixels, gray p99=>210, gain<=64, gamma0.8',
                             local='exposed original, Lab-L CLAHE clip1.8 grid8x8',
                             detail='supplied bright, bilateral d5/color18/space3; unsharp0.6 sigma0.9',
                             retinex='original gray; illumination estimated at longside<=768 with log Gaussian15/80; p1-p99 stretch',
                             expanded='supplied bright, native crop margin enlarged by max32px/15% of shorter crop side'),
                source_bright_sha256=fingerprints,source_sha256=sha256(Path(__file__)),
                scope='pilot human viewing only; no OCR, model input, transcription or source pixel changes')
    (package/'human_views_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if binding != sha256(package/'protocol.json'):
        raise ValueError('Frozen package identity changed')
    render_pages(package)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path('data/derived/v5/lated_mvp_20261026'))
    parser.add_argument('--auto-only',action='store_true',help='Select human viewing presets for the existing pilot')
    args=parser.parse_args();result=recommend(args.package) if args.auto_only else prepare(args.package)
    print(json.dumps({k:result[k] for k in ('status','images','instances') if k in result}),flush=True)
