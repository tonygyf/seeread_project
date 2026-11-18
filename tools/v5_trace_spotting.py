"""Read-only stage inspection and paired development detection diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def page(path, title, body):
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>'
        'body{font:17px system-ui;background:#15232f;color:#edf2f7;margin:24px auto;padding:0 20px;max-width:1400px;line-height:1.6}'
        'a{color:#85ceff}img{max-width:100%}table{border-collapse:collapse;width:100%}'
        'td,th{padding:9px;border-bottom:1px solid #536674;text-align:left}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}'
        '.batch{display:flex;min-width:1050px}.labels{width:280px;flex-shrink:0}'
        '.label{height:384px;box-sizing:border-box;padding:14px;overflow:auto;border-bottom:1px solid #8298a6}'
        '.batch img{width:768px;max-width:none;height:auto;align-self:start;filter:brightness(var(--view-gain,1))}'
        'figure{margin:0}small{color:#bbcad5}input{font-size:18px;padding:8px;width:90%}'
        '</style>' + f'<h1>{html.escape(title)}</h1>' + body + '</html>')


def save_rgb(path, pixels):
    import cv2
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def check_manifest(manifest, parent_report):
    if hashlib.sha256(Path(manifest).read_bytes()).hexdigest()!=parent_report['manifest_sha256']:
        raise ValueError('Trace GT manifest differs from the scored manifest')


def view_controls():
    return '<p>仅显示提亮，不改变模型输入或预测：<input aria-label="裁剪显示亮度" type="range" min="1" max="16" step="0.25" value="1" oninput="document.documentElement.style.setProperty(\'--view-gain\',this.value);this.nextElementSibling.value=this.value+\'×\'"><output>1×</output>。左右裁剪使用相同显示增益。</p>'


def crop_atlas(directory, name, pairs):
    """Keep each 384px crop intact; put raw/enhanced copies in adjacent columns."""
    import numpy as np
    if not pairs:
        return []
    names = []
    for start in range(0, len(pairs), 16):
        batch = pairs[start:start + 16]
        tiles = []
        for raw, enhanced in batch:
            if raw.shape != (384, 384, 3) or enhanced.shape != raw.shape:
                raise ValueError('Stage atlas requires exact 384x384 RGB crop pairs')
            tiles.append(np.concatenate((raw, enhanced), axis=1))
        filename = f'{name}_{start:04d}.png'
        save_rgb(directory / filename, np.concatenate(tiles, axis=0))
        names.append((filename, start, len(batch)))
    return names


def compare(config, out):
    import torch
    from torch.nn import functional as F
    from seeread.v5.seeing import build_detector
    from seeread.v5.seeing_decode import decode
    from seeread.v5.lated_evaluation import image_counts, summary
    from seeread.v5.reading import sha256
    from tools.v5_eval_spotting import verified_rows, rgb_tensor
    rows = verified_rows(config)
    report = dict(status='running', scope='paired development only; fixed crop-free detection protocol',
                  input_hw=config['detector_hw'], models={})
    for label, checkpoint in config['checkpoints'].items():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_detector(saved['config']['detector'])
        model.load_state_dict(saved['model'], strict=True)
        model = model.requires_grad_(False).eval().cuda()
        result = dict(checkpoint=checkpoint, sha256=sha256(checkpoint), state=saved['state'], results={})
        del saved
        for light, samples in rows.items():
            total, foreground = Counter(), []
            for row in samples:
                image = rgb_tensor(Path(config['data_root']) / row['image'], 'cuda:0')
                prediction = model(F.interpolate(image[None], size=tuple(config['detector_hw']), mode='bilinear', align_corners=False))
                detections = decode(prediction, image_hw=image.shape[-2:], **config['decoder'])
                counts, _ = image_counts(detections, row['instances']); total.update(counts)
                foreground.append({key:float((prediction[key].softmax(1)[:, 1] >= .5).float().mean()) for key in ('text','center')})
            result['results'][light] = dict(images=len(samples), **summary(total),
                mean_foreground_fraction={k:sum(v[k] for v in foreground)/len(foreground) for k in ('text','center')})
            print(json.dumps(dict(checkpoint=label, illumination=light, **result['results'][light])), flush=True)
        report['models'][label] = result
        write_json(out/'run.json', report)
        del model
        torch.cuda.empty_cache()
    report['status'] = 'complete'; write_json(out/'run.json', report)


def trace(config, out):
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tools import v5_lated
    from tools.v5_eval_spotting import rgb_tensor
    from seeread.v5.spotting import polygon_crop, crop_exposure
    from seeread.v5.seeing_decode import decode
    from seeread.v5.lated_annotations import read_jsonl, latin_text
    from seeread.v5.lated_evaluation import image_counts
    from seeread.v5.reading import sha256
    args = argparse.Namespace(detector=Path(config['detector']), reading=Path(config['reading']), max_len=128)
    spotter, decoder, _ = v5_lated.models(args)
    report = dict(status='running', scope='read-only visualization of completed inference; no new model selection',
        detector_sha256=sha256(args.detector), reading_sha256=sha256(args.reading), cases={})
    index = []
    rgb = lambda tensor: tensor.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    for case in config['cases']:
        parent = Path(case['run']); previous = json.loads((parent/'run.json').read_text())
        for key in ('detector_sha256','reading_sha256'):
            if previous[key] != report[key]: raise ValueError('Trace weights differ from completed inference')
        if previous['crop'] != json.loads(json.dumps(spotter.crop_kwargs)) or previous['decoder'] != decoder:
            raise ValueError('Trace geometry differs from completed inference')
        check_manifest(case['manifest'],previous)
        rows, records = read_jsonl(case['manifest']), read_jsonl(parent/'records.jsonl')
        if [r['image'] for r in rows] != [r['image'] for r in records] or previous['status'] != 'complete' or len(records)!=previous['images']:
            raise ValueError('Trace requires complete ordered source records')
        directory = out/case['id']; directory.mkdir()
        for number, (row, record) in enumerate(zip(rows, records)):
            scene = directory/f'{number:04d}'; scene.mkdir()
            image = rgb_tensor(Path(config['data_root'])/row['image'], 'cuda:0')
            if sha256(Path(config['data_root'])/row['image']) != previous['image_sha256'][row['image']]:
                raise ValueError('Trace input image differs from scored input')
            canvas = F.interpolate(image[None], size=tuple(previous['detector_hw']), mode='bilinear', align_corners=False)
            prediction = spotter.detector(canvas)
            replay = decode(prediction,image_hw=image.shape[-2:],**decoder)
            if len(replay)!=len(record['predictions']) or any(
                    np.asarray(a['poly']).shape!=np.asarray(b['poly']).shape or
                    not np.allclose(a['poly'],b['poly'],atol=1e-3,rtol=0) or abs(a['score']-b['score'])>1e-5
                    for a,b in zip(replay,record['predictions'])):
                raise ValueError('Replayed See output differs from scored polygons')
            save_rgb(scene/'input.png', rgb(canvas[0]))
            maps = {k:prediction[k].softmax(1)[0,1].cpu().numpy() for k in ('text','center')}
            maps.update({k:prediction[k][0,0].cpu().numpy() for k in ('height','angle')})
            np.savez_compressed(scene/'see_maps.npz', **maps)
            for key in ('text','center'):
                heat = cv2.applyColorMap(np.rint(maps[key]*255).astype(np.uint8), cv2.COLORMAP_TURBO)
                cv2.imwrite(str(scene/f'{key}.png'), cv2.resize(heat, (640,640), interpolation=cv2.INTER_NEAREST))
            ground = [dict(g,ocr_eligible=not g.get('ignore') and latin_text(g.get('text','')) and g.get('text')!='###') for g in row['instances']]
            counts, matches = image_counts(record['predictions'],ground)
            matched = dict(matches); used = {gi for _,gi in matches}
            overlay = rgb(image).copy()
            for gi,g in enumerate(ground):
                color=(150,150,150) if g.get('ignore') else ((80,190,255) if gi in used else (255,160,30))
                cv2.polylines(overlay,[np.rint(g['poly']).astype(np.int32)],True,color,2)
                x,y=np.rint(g['poly'][0]).astype(int)
                cv2.putText(overlay,f'G{gi}',(int(x),max(12,int(y)-12)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            for pi,p in enumerate(record['predictions']):
                poly=np.rint(p['poly']).astype(np.int32)
                cv2.polylines(overlay,[poly],True,(50,255,80) if pi in matched else (255,65,100),2)
                cv2.putText(overlay,f'P{pi}',tuple(poly[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            save_rgb(scene/'overlay.png',overlay)
            paired = Path(config['data_root'])/row.get('lit',row['image'])
            if paired != Path(config['data_root'])/row['image']:
                save_rgb(scene/'reference.jpg',rgb(rgb_tensor(paired,'cpu')))
            groups=[]
            for kind,items in [('pred',record['predictions']),('gt',ground)]:
                pairs, labels, detail = [], [], []
                for i,item in enumerate(items):
                    result=polygon_crop(image,item['poly'],**spotter.crop_kwargs)
                    title=f'P{i}' if kind=='pred' else f'G{i} · GT {item.get("id",i)}'
                    text=item.get('text','') if kind=='pred' else (item.get('text','') if item['ocr_eligible'] else '无可用转写真值')
                    note=f'{title}\n文字：{text}'
                    if kind=='pred':
                        gi=matched.get(i); note+=f'\n检测分数：{item["score"]:.3f}\n'+(f'匹配GT：{ground[gi].get("id",gi)}\nGT：{ground[gi]["text"] if ground[gi]["ocr_eligible"] else "待转写"}' if gi is not None else '未达到一对一IoU>0.5匹配')
                        if result.reason != item.get('crop_error'): raise ValueError('Trace crop failure differs from scored output')
                    elif case['id']!='lated':
                        saved=next((c for c in record['gt_crops'] if c['id']==item.get('id')),None)
                        if saved: note+=f'\nGT裁剪识别：{saved["text"]}'
                    note+=f'\n裁剪：{result.reason or result.sampling}\n回退原因：{result.fallback_reason or "无"}'
                    if result.crop is None:
                        raw=enhanced=np.zeros((384,384,3),np.uint8)
                    else:
                        adjusted,_=crop_exposure(result.crop,spotter.exposure)
                        raw,enhanced=rgb(adjusted[0]),rgb(spotter.reading.enhancer(adjusted)[0])
                    pairs.append((raw,enhanced)); labels.append(note)
                    detail.append(dict(index=i,kind=kind,text=text,crop_error=result.reason,crop_strategy=result.sampling,
                                       centers=result.centers.tolist() if result.centers is not None else None))
                batches=[]
                for filename,start,count in crop_atlas(scene,kind,pairs):
                    descriptions=''.join('<div class="label">'+html.escape(v).replace('\n','<br>')+'</div>' for v in labels[start:start+count])
                    batches.append(f'<div style="overflow-x:auto"><div class="batch"><div class="labels">{descriptions}</div><img loading="lazy" src="{filename}" alt="左：实际展开裁剪；右：Self-DACE实际输出"></div></div>')
                groups.append('<h2>'+('预测轮廓裁剪 → Self-DACE → Read文字' if kind=='pred' else 'GT轮廓裁剪对照')+'</h2><p>每行左图为384×384实际裁剪，右图为Self-DACE实际输出。识别文字复用已完成推理记录。</p>'+''.join(batches))
                write_json(scene/f'{kind}.json',detail)
            nav=f'<p><a href="../../index.html">返回全部图像</a> · {html.escape(row["image"])}</p>'
            if number: nav+=f'<a href="../{number-1:04d}/index.html">上一张</a> · '
            if number+1<len(rows): nav+=f'<a href="../{number+1:04d}/index.html">下一张</a>'
            body=nav+'<p>当前所选原图进入See；GT仅用于事后核对。SCM是训练辅助分支，推理时不运行。Read输入为右侧增强图，再归一化到[-1,1]；score仅表示检测置信度。</p>'+view_controls()
            figures=[('实际See输入（640×640）','input.png'),('See文字概率（蓝低、红高，固定0–1）','text.png'),('See中心线概率（蓝低、红高，固定0–1）','center.png'),('原图坐标下轮廓：绿预测匹配、红预测未匹配；蓝GT匹配、橙GT漏检、灰ignore','overlay.png')]
            if (scene/'reference.jpg').exists(): figures.append(('配对亮图，仅作人眼参考，不进入暗图推理','reference.jpg'))
            body+='<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(t)}</figcaption><a href="{f}"><img src="{f}"></a></figure>' for t,f in figures)+'</div>'
            body+='<p><a href="see_maps.npz">原始See概率／高度／角度数组</a></p>'+''.join(groups)
            page(scene/'index.html',f'{case["id"]} · {number+1}/{len(rows)} · 逐阶段核对',body)
            index.append(f'<tr><td>{case["id"]}</td><td><a href="{case["id"]}/{number:04d}/index.html">{html.escape(row["image"])}</a></td><td>{len(record["predictions"])}</td><td>{counts.get("tp_det",0)}</td><td>{counts["n_gt"]-counts.get("tp_det",0)}</td></tr>')
            print(json.dumps(dict(case=case['id'],images=number+1,expected=len(rows))),flush=True)
        report['cases'][case['id']]=dict(images=len(rows),parent_report_sha256=sha256(parent/'run.json'),
            manifest_sha256=sha256(case['manifest']),records_sha256=sha256(parent/'records.jsonl'))
        write_json(out/'run.json',report)
        page(out/'index.html','See → connector → Read 逐阶段核对','<p>查看固定基线实际中间图；所有测试结果保持不变。点击图像进入热图、轮廓、预测／GT裁剪与增强对照。数值配方审计见主文档04。</p><input placeholder="搜索文件名或数据集" oninput="for(const r of document.querySelectorAll(\'tbody tr\'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())"><table><thead><tr><th>数据</th><th>图像</th><th>预测框数</th><th>检测匹配</th><th>漏检GT</th></tr></thead><tbody>'+''.join(index)+'</tbody></table>')
    report['status']='complete'; write_json(out/'run.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    from tools.v5_lated import bootstrap
    bootstrap(Path(config['runtime']))
    import torch
    import cv2
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Use one authorized idle GPU container')
    torch.set_num_threads(4); cv2.setNumThreads(4); torch.cuda.set_per_process_memory_fraction(.35)
    out=Path(config['output']); out.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        {'trace':trace,'compare':compare}[config['mode']](config,out)


if __name__=='__main__':
    main()
