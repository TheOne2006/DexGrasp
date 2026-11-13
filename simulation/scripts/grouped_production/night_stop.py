"""Request this owned run to stop new Init, drain, and exit within a finite grace."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--grace-seconds',type=int,required=True);a=p.parse_args();assert 1<=a.grace_seconds<=3600
c=json.loads((a.run/'command.json').read_text())['config'];assert c.get('checkpoint_schema')=='floating-night/v2'
pid=json.loads((a.run/'process.json').read_text());assert Path(f'/proc/{pid["pid"]}/stat').read_text().split(')')[-1].split()[19]==pid['start_ticks'],'Owned process has exited or PID was reused'
assert not (a.run/'stop-request.json').exists(),'Do not overwrite an existing stop request'
request={'requested_epoch':time.time(),'hard_deadline_epoch':time.time()+a.grace_seconds,'pid':pid['pid'],'start_ticks':pid['start_ticks'],'reason':'explicit owned-rank drain request'};tmp=a.run/'stop-request.partial';tmp.write_text(json.dumps(request));os.replace(tmp,a.run/'stop-request.json');print(request)
