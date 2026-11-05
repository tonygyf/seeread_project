"""Frozen R4 diagnostic on human-reviewed subsets, preserving original split labels."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from seeread.v5.reading import WordCropDataset,sha256
from seeread.v5.lated_annotations import read_jsonl
from seeread.v5.scene_str import score_catalog
from seeread.v5.recognition import build_pipeline
from tools.v5_reading import evaluate,source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--arm',type=Path,action='append',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-len',type=int,default=128)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One idle authorized GPU is required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35,0)
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    if not parent.get('resume_safe') or parent['source_hashes']!=source_hashes():raise ValueError('Parent is unhealthy or Reading sources changed')
    cfg=copy.deepcopy(parent['config']);cfg['recognizer']['max_len']=args.max_len
    cfg['enhancer']['freeze']=True;cfg['recognizer']['trainability']='frozen'
    model=build_pipeline(cfg);model.load_state_dict(parent['model'],strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    if args.max_len>model.recognizer.base.decoder.config.max_position_embeddings:raise ValueError('Decoder capacity exceeded')
    texts=[r['text'] for arm in args.arm for split in ('train','dev') for r in read_jsonl(arm/f'{split}.jsonl')]
    lengths=[len(model.recognizer.tok(t).input_ids) for t in texts]
    if not lengths or max(lengths)>args.max_len:raise ValueError('Empty review subset or label exceeds shared generation cap')
    model.eval().to('cuda:0');args.output.mkdir(parents=True)
    report=dict(status='running',scope='frozen GT-crop recognition; source-train-split diagnostic before LATeD adaptation; any dev subset is partial and human-selected',
                parent=str(args.parent),parent_sha256=sha256(args.parent),source_hashes=source_hashes(),max_len=args.max_len,
                max_label_tokens=max(lengths),weights_updated=False,arms={})
    started=time.monotonic()
    def publish():
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    publish()
    for arm in args.arm:
        results={};out=args.output/arm.name;out.mkdir()
        protocol=json.loads((arm/'protocol.json').read_text())
        for split in ('train','dev'):
            catalog=read_jsonl(arm/f'{split}.catalog.jsonl')
            if (arm/f'{split}.jsonl').stat().st_size:
                dataset=WordCropDataset(arm/f'{split}.jsonl',args.root,split)
                result=evaluate(model,dataset,'cuda:0',4,out/f'{split}.records.jsonl')
                predictions=read_jsonl(out/f'{split}.records.jsonl')
                if not result['complete']:raise ValueError('Partial diagnostic evaluation')
            else:predictions=[]
            results[split]=score_catalog(catalog,predictions)
            results[split]['evaluation_role']='train_sample_diagnostic' if split=='train' else 'partial_development_diagnostic'
            print(json.dumps(dict(arm=arm.name,split=split,result=results[split]),ensure_ascii=False),flush=True)
        report['arms'][arm.name]=dict(crop_method=protocol['method'],sidecar_sha256=protocol['sidecar_sha256'],results=results);publish()
    report['status']='complete';publish()


if __name__=='__main__':main()
