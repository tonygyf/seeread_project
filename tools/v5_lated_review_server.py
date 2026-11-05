"""Serve the existing review page and persist explicit browser review snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReviewStore:
    def __init__(self, packages):
        self.packages = {}
        self.lock = threading.Lock()
        for directory in packages:
            directory = Path(directory)
            tasks = {r['id']: r for line in (directory/'tasks.jsonl').open() if (r:=json.loads(line))}
            self.packages[sha256(directory/'protocol.json')] = directory, tasks

    def save(self, payload):
        binding = payload.get('protocol_sha256')
        if binding not in self.packages:
            raise ValueError('Unknown review protocol')
        directory, tasks = self.packages[binding]
        annotations = payload.get('annotations')
        if not isinstance(annotations, list) or not isinstance(payload.get('reviewer', ''), str):
            raise ValueError('Invalid review snapshot')
        seen = set()
        for row in annotations:
            identity = row.get('id')
            if identity not in tasks or identity in seen:
                raise ValueError('Unknown or duplicate review ID')
            seen.add(identity)
            if row.get('status') not in ('pending','reviewed','unreadable') or row.get('language') not in ('unknown','latin','non_latin','mixed') or not isinstance(row.get('text'),str):
                raise ValueError('Invalid annotation fields')
        with self.lock:
            out = directory/'review_sync'
            out.mkdir(exist_ok=True)
            text = json.dumps(payload,ensure_ascii=False,indent=2)+'\n'
            # Immutable content snapshots preserve prior states if a browser later clears labels.
            content = hashlib.sha256(text.encode()).hexdigest()
            snapshot = out/(content+'.json')
            if not snapshot.exists():snapshot.write_text(text)
            temp = out/'latest.json.tmp';temp.write_text(text);temp.replace(out/'latest.json')
            counts = {s:sum(r['status']==s for r in annotations) for s in ('reviewed','unreadable','pending')}
            summary = dict(updated_utc=datetime.now(timezone.utc).isoformat(),counts=counts,
                           reviewed_total=counts['reviewed']+counts['unreadable'],reviewer=payload.get('reviewer',''),snapshot=snapshot.name)
            (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
            for old in sorted((p for p in out.glob('*.json') if len(p.stem)==64),key=lambda p:p.stat().st_mtime,reverse=True)[20:]:
                old.unlink()

        return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--package',type=Path,action='append',required=True)
    p.add_argument('--bind',default='100.88.81.10')
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args();store=ReviewStore(args.package)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(args.root),**kw)
        def do_POST(self):
            if self.path!='/api/reviews':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16*1024**2:raise ValueError('Review body exceeds16MiB or is empty')
                origin=self.headers.get('Origin')
                if origin and origin.rstrip('/')!='http://'+self.headers.get('Host',''):
                    raise ValueError('Cross-origin review writes are refused')
                payload=json.loads(self.rfile.read(length));result=store.save(payload)
                body=json.dumps(result,ensure_ascii=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:
                self.send_error(400,str(e))
    ThreadingHTTPServer((args.bind,args.port),Handler).serve_forever()


if __name__=='__main__':main()
