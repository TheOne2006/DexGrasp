import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class Configure(unittest.TestCase):
    def test_full_shard_six_scales_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base.json';index=root/'index.json';out=root/'ranks'
            base.write_text(json.dumps({'hard_deadline_epoch':123,'tours':2,'max_group_visits':2,'case_limit_s':5,'stop_after_jobs':2}))
            cases=[dict(object_id=f'obj{i:04d}',scene_id=f'obj{i:04d}_scale{s}') for i in range(5697) for s in ['005','008','011','014','017','020']]
            index.write_text(json.dumps({'cases':cases}))
            subprocess.run([sys.executable,str(Path(__file__).with_name('night_configure.py')),str(base),str(index),str(out),'--template-only','--continuous','--scales-per-group','6'],check=True,capture_output=True)
            m=json.loads((out/'sharding.json').read_text())
            self.assertEqual(sorted(len(x['objects']) for x in m['assignments']),[712]*7+[713])
            self.assertEqual(m['configurations'],34182)
            union=set()
            for a in m['assignments']:
                self.assertFalse(union.intersection(a['objects']));union.update(a['objects'])
                c=json.loads(Path(a['config']).read_text())
                for key in ['tours','hard_deadline_epoch','stop_new_init_epoch','max_group_visits']:self.assertIsNone(c[key])
                self.assertNotIn('case_limit_s',c);self.assertNotIn('stop_after_jobs',c)
                sizes=[len(json.loads(Path(g['index']).read_text())['cases']) for g in c['groups']]
                self.assertEqual(sizes[:7],[600]*7)
                self.assertEqual(sum(sizes),a['configurations'])
            self.assertEqual(len(union),5697)

if __name__=='__main__':unittest.main()
