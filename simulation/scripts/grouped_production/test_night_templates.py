import hashlib,json,tempfile,unittest
from pathlib import Path
from night_templates import install_snapshot

class Templates(unittest.TestCase):
    def test_install_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);snap=root/'snapshot';snap.mkdir();out=root/'checkpoint';out.mkdir()
            assets=root/'assets/hand/sharpa_v2/init_tmpl';assets.mkdir(parents=True)

            for i in range(34):(assets/f'seed{i}.npy').write_bytes(b'seed')
            (assets/'family.npy').write_bytes(b'seed');(snap/'family.npy').write_bytes(b'opaque-test-bundle')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            m=dict(status='PASS',seed_count=35,seed_hashes={p.stem:sha(p) for p in assets.glob('*.npy')},bundle_hashes={'family.npy':sha(snap/'family.npy')},content_unique=1,references={'family':[dict(id='one',path='seed-history/family.npy',index=0)]})
            (snap/'manifest.json').write_text(json.dumps(m))
            c=dict(template_snapshot=str(snap),template_manifest_sha256=sha(snap/'manifest.json'),asset_root=str(root/'assets'))
            s=dict(completed=[],family_archive={});install_snapshot(c,s,out)
            self.assertEqual(s['feedback_seen_ids'],['one']);self.assertEqual(s['family_archive'],m['references'])
            s['completed'].append('job');install_snapshot(c,s,out)
            self.assertEqual(s['feedback_seen_ids'],['one'])
            (snap/'manifest.json').write_text('{}')
            with self.assertRaises(AssertionError):install_snapshot(c,s,out)

if __name__=='__main__':unittest.main()
