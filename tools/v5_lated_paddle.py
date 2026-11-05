"""PP-OCRv5 drafts on frozen LATeD GT crops; every confidence retained for review."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from seeread.v5.lated_annotations import read_jsonl, render_pages
from seeread.v5.reading import sha256


def balance(image):
    low, high = np.percentile(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (1, 99.5))
    return np.round(np.clip((image.astype(np.float32)-low)/max(float(high-low), 3.), 0, 1)**.85*255).astype(np.uint8)


def initialize_package(package, previous):
    protocol_path = package/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    tasks = read_jsonl(package/'tasks.jsonl')
    if sha256(package/'tasks.jsonl') != protocol['tasks_sha256']:
        raise ValueError('GT task manifest changed')
    if protocol['images'] != {'train': 900, 'dev': 100}:
        raise ValueError('Require the fixed 900/100 package; test is excluded')
    if previous and 'legacy_review_protocols' not in protocol:
        old = read_jsonl(previous/'tasks.jsonl'); new = {t['id']: t for t in tasks}
        for task in old:
            if task['id'] not in new or any(task[k] != new[task['id']][k] for k in ('image','image_sha256','poly','split')):
                raise ValueError('Old review cannot be migrated to different geometry/split')
        protocol['legacy_review_protocols'] = [sha256(previous/'protocol.json')]
        protocol['review_migration'] = 'Exact same instance/image digest/polygon/split; browser keeps reviewed text/status by ID'
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+'\n')
        for filename in ('drafts.jsonl','review_assists.jsonl','human_views.jsonl','quality_hints.jsonl'):
            source = previous/filename
            if source.exists():
                (package/filename).write_bytes(source.read_bytes())
        for directory in ('crops_lle','human_views','human_auto_selected'):
            if (previous/directory).exists():
                (package/directory).symlink_to(Path('../')/previous.name/directory, target_is_directory=True)
    return protocol, tasks


def strip_view(bright_crop, polygon):
    # Reuse the existing connector's GT-polygon sampler, on CPU only.
    import torch
    from seeread.v5.spotting import polygon_crop
    image = torch.from_numpy(cv2.cvtColor(bright_crop, cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)
    first = polygon_crop(image, polygon, output_hw=(64,512), pad=1.1, sampling='boundary_when_supported')
    if first.crop is None:
        return None, first.reason
    length = float(np.linalg.norm(np.diff(first.centers,axis=0),axis=1).sum())
    height = max(float(np.mean(first.heights)),1.)
    width = min(3200,max(64,round(64*length/height)))
    result = polygon_crop(image, polygon, output_hw=(64,width), pad=1.1, sampling='boundary_when_supported')
    if result.crop is None:
        return None, result.reason
    rgb = result.crop[0].permute(1,2,0).mul(255).round().byte().numpy()
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), None


def run(args):
    import paddle
    from paddleocr import TextRecognition
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count()!=1:
        raise RuntimeError('Require one visible CUDA GPU; CPU fallback is forbidden')
    paddle.set_device('gpu:0')
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    package=args.package.resolve()
    protocol,tasks=initialize_package(package,args.previous.resolve() if args.previous else None)
    root=Path(protocol['data_root']); binding=sha256(package/'protocol.json')
    output=package/'paddle_drafts.jsonl'
    done={r['id'] for r in read_jsonl(output)} if output.exists() else set()
    if len(done)>len(tasks):raise ValueError('Existing drafts contain unknown tasks')
    records_path=package/'paddle_drafts_provenance.json'
    model_hashes={p.name:sha256(p) for p in args.model.glob('inference.*') if p.is_file()}
    report=dict(status='running',model='PP-OCRv5_server_rec',model_revision='b26c3587fda8da3c8ec0ce357214b4d661ff1558',
                model_hashes=model_hashes,package_sha256=binding,source_sha256=sha256(__file__),
                paddle=paddle.__version__,gpu=paddle.device.cuda.get_device_name(0),
                expected=len(tasks),completed=len(done),images_expected=1000,
                scope='machine drafts only; original GT regions; never ground truth',
                confidence_policy='keep every score including zero/empty; uncalibrated candidate ranking only',
                bright_reference='dataset-supplied aligned enhancement for annotation assistance; not an inference input assumption',
                rotations=[0,90,180,270],threshold=None)
    if records_path.exists():
        old=json.loads(records_path.read_text())
        for k in ('model_hashes','package_sha256','source_sha256'):
            if old[k]!=report[k]:raise ValueError('Resume provenance differs: '+k)
    started=time.monotonic()
    def publish():
        report['elapsed_seconds']=time.monotonic()-started
        report['instances_per_second']=(report['completed']-len(done))/max(report['elapsed_seconds'],1e-6)
        report['remaining_seconds_estimate']=(len(tasks)-report['completed'])/report['instances_per_second'] if report['instances_per_second']>0 else None
        temporary=records_path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');temporary.replace(records_path)
        render_pages(package)
    publish()
    model=TextRecognition(model_name='PP-OCRv5_server_rec',model_dir=str(args.model),device='gpu:0')
    sources={r['image']:r for source in protocol['sources'].values() for r in read_jsonl(source['path'])}
    groups=defaultdict(list)
    for task in sorted(tasks,key=lambda t:(not t['pilot'], t['split'],t['image'],t['instance_index'])):
        if task['id'] not in done:groups[task['image']].append(task)
    (package/'ocr_bright').mkdir(exist_ok=True)
    (package/'ocr_rectified').mkdir(exist_ok=True)
    human_path=package/'paddle_human_views.jsonl'
    with output.open('a') as stream,human_path.open('a') as human_stream:
        pending=[]; pending_meta=[]; candidates=defaultdict(list); task_meta={}
        def infer_pending():
            if not pending:return
            result=list(model.predict(pending,batch_size=args.batch_size))
            if len(result)!=len(pending_meta):raise ValueError('OCR output count differs from candidates')
            for record,meta in zip(result,pending_meta):
                score=float(record['rec_score'])
                if not math.isfinite(score):raise ValueError('Nonfinite OCR confidence')
                candidates[meta['id']].append(dict(view=meta['view'],rotation=meta['rotation'],text=str(record['rec_text']),confidence=score))
            pending.clear();pending_meta.clear()
        for scene,items in groups.items():
            row=sources[scene]
            bright_path=root/row['lit']
            bright=cv2.imread(str(bright_path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            original=cv2.imread(str(root/scene),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if bright is None or original is None or bright.shape!=original.shape:
                raise ValueError('Aligned bright/original geometry missing: '+scene)
            h,w=bright.shape[:2]
            for task in items:
                polygon=np.asarray(task['poly'],np.float32)
                x0,y0=np.maximum(np.floor(polygon.min(0)-8).astype(int),0)
                x1,y1=np.minimum(np.ceil(polygon.max(0)+8).astype(int),[w,h])
                raw=original[y0:y1,x0:x1]
                supplied=balance(bright[y0:y1,x0:x1])
                target=package/'ocr_bright'/Path(task['crop']).name
                if not cv2.imwrite(str(target),supplied):raise OSError(target)
                strip,error=strip_view(supplied,polygon-[x0,y0])
                views={'original':raw,'dataset_bright':supplied}
                if strip is not None:
                    views['rectified_bright']=strip
                    if not cv2.imwrite(str(package/'ocr_rectified'/target.name),strip):raise OSError(target)
                task_meta[task['id']]=dict(rectification_error=error)
                human_stream.write(json.dumps(dict(id=task['id'],views={'balanced':str(target.relative_to(package)),'bright':str(target.relative_to(package))},
                    bbox=[int(x0),int(y0),int(x1),int(y1)],poly_local=(polygon-[x0,y0]).tolist(),context=task['context'],human_aid_only=True))+'\n')
                for view,pixels in views.items():
                    angles=(0,180) if view=='rectified_bright' else (0,90,180,270)
                    for angle in angles:
                        rotated=np.ascontiguousarray(np.rot90(pixels,angle//90))
                        # Recognition preprocessing resizes to 48px high; bound source memory for very large GT lines.
                        factor=min(1.,2400/max(rotated.shape[:2]))
                        if factor<1:rotated=cv2.resize(rotated,(max(2,round(rotated.shape[1]*factor)),max(2,round(rotated.shape[0]*factor))))
                        pending.append(rotated);pending_meta.append(dict(id=task['id'],view=view,rotation=angle))
                        if len(pending)>=args.batch_size:infer_pending()
            infer_pending()
            for task in items:
                options=candidates.pop(task['id'])
                best=max(options,key=lambda c:c['confidence'])
                stream.write(json.dumps(dict(id=task['id'],text=best['text'],confidence=best['confidence'],best=best,candidates=options,
                    status='machine_draft_not_ground_truth',**task_meta.pop(task['id'])),ensure_ascii=False)+'\n')
                report['completed']+=1
            stream.flush();human_stream.flush()
            if report['completed']%200 < len(items) or report['completed']==len(tasks):
                publish();print(json.dumps({k:report[k] for k in ('completed','expected','elapsed_seconds','remaining_seconds_estimate')}),flush=True)
    report['status']='complete';report['source_images']=len({t['image'] for t in tasks})
    publish();print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--model',type=Path,default=Path('/models/recognizer'))
    parser.add_argument('--batch-size',type=int,default=32)
    run(parser.parse_args())
