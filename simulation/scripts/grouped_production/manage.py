"""Run or inspect the floating-night/v2 production entry."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
r=sub.add_parser('run');r.add_argument('config',type=Path);r.add_argument('--name',required=True)
for action in ('stats','verify'):sub.add_parser(action).add_argument('run',type=Path)
a=p.parse_args();scripts=Path(__file__).resolve().parent
if a.action=='run':cmd=[sys.executable,str(scripts/'run.py'),str(a.config),'--name',a.name]
else:
 c=json.loads((a.run/'command.json').read_text())['config']
 if c.get('worker')!='night_worker.py':p.error('This entry supports night_worker.py runs only')
 if a.action=='verify':cmd=[c['python'],'-B',str(scripts/'night_verify.py'),c['checkpoint_root'],'--output',str(a.run/'verification.json'),'--family-capacity',str(c.get('family_capacity',512))]
 else:cmd=[c['python'],'-B',str(scripts/'vnext_stats.py'),str(a.run)]
raise SystemExit(subprocess.call(cmd))
