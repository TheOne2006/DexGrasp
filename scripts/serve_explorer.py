"""Serve only the built explorer and whitelisted, read-only experiment records.
Bind localhost; the host's existing Tailscale userspace network provides access.
"""
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import re
import numpy as np

BASE=Path(__file__).resolve().parents[2]
DATA=BASE/'data/sharpa_dexonomy_1k_v1'
OUTPUT=BASE/'outputs/sharpa_dexonomy_1k'
SITE=Path(__file__).resolve().parents[1]/'site/dist'

def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(SITE),**kwargs)
    def do_GET(self):
        path=self.path.split('?',1)[0]
        if not path.startswith('/api/'):
            return super().do_GET()
        try:
            if path=='/api/summary':
                result=read_json(DATA/'preparation.json',{'status':'preparing'})
                result['independent_fk']=read_json(OUTPUT/'independent_fk.json')
                result['software_tests']=read_json(OUTPUT/'software_tests.json')
            elif path=='/api/records':result=read_json(DATA/'records.json',[])
            elif path=='/api/protocol':result=read_json(OUTPUT/'protocol.json',{})
            elif path=='/api/runs':
                result=[]
                for folder in sorted(OUTPUT.iterdir()):
                    log=folder/'metrics.jsonl'
                    if not folder.is_dir() or not log.exists():continue
                    rows=[]
                    for line in log.read_text().splitlines():
                        try:rows.append(json.loads(line))
                        except json.JSONDecodeError:pass # writer may be appending
                    cfg=read_json(folder/'provenance.json',{}).get('config',{})
                    last=rows[-1]['step'] if rows else 0
                    values=[r['train_reconstruction']['mpjpe_mm'] for r in rows if 'train_reconstruction' in r and r['step']>0]
                    result.append(dict(name=folder.name,metrics=[r for r in rows if 'train_reconstruction' in r],latest_step=last,
                                       best_mpjpe=min(values) if values else None,finished=last>=cfg.get('max_steps',float('inf')),
                                       evaluation=read_json(folder/'evaluation.json')))
            elif re.fullmatch(r'/api/sample/[a-f0-9]{20}',path):
                gid=path.rsplit('/',1)[-1]
                records=read_json(DATA/'records.json',[])
                record=next((r for r in records if r['id']==gid),None)
                if record is None:return self.send_error(404)
                with np.load(DATA/f'assets/{gid}.npz',allow_pickle=False) as mesh:
                    result={k:mesh[k].tolist() for k in ('vertices','faces')}
                with np.load(DATA/f'raw_grasps/{gid}.npz',allow_pickle=False) as raw:
                    result.update({k:raw[k].tolist() for k in ('K','shadow_K','q','R','t','source_qpos')})
                result['record']=record
                result['predictions']={}
                for method in ('fm','flowmap'):
                    prediction=OUTPUT/method/'site_predictions.npz'
                    if not prediction.exists():continue
                    with np.load(prediction,allow_pickle=False) as predictions:
                        if str(predictions['method'])!=method:
                            raise ValueError('Prediction method does not match run')
                        matches=np.flatnonzero(predictions['ids']==gid)
                        if len(matches)==1:
                            result['predictions'][method]={
                                'keypoints':predictions['keypoints'][matches[0]].tolist(),
                                'seed':int(predictions['seed']),
                                'checkpoint_sha256':str(predictions['checkpoint']),
                            }
            else:return self.send_error(404)
            payload=json.dumps(result,allow_nan=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        except (OSError,ValueError,KeyError) as e:
            self.log_error('%s',e);self.send_error(500,'Experiment data unavailable')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8766);args=p.parse_args()
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
