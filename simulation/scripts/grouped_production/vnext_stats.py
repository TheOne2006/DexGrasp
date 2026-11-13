"""Report saved progress without counting unfinished stages as production."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();c=json.loads((a.run/'command.json').read_text())['config'];root=Path(c.get('checkpoint_root',a.run/'data'));state=np.load(root/'state.npy',allow_pickle=True).item();rows=state['rows'];keys=['init','refined','stable','feedback','init_wall_s','refine_wall_s','eval_wall_s','prepare_s'];result={'completed_jobs':len(rows),'planned_jobs':state.get('planned_jobs'),'totals':{k:sum(r[k] for r in rows) for k in keys},'checkpoint_root':str(root),'completion':'COMPLETE' if len(rows)==state.get('planned_jobs') else 'PARTIAL'}
(a.run/'progress-stats.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
