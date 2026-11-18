"""Opt-in real CUDA contact regression; no physical grasp success claim."""
import os
import unittest

@unittest.skipUnless(os.environ.get('DEXONOMY_GPU_TESTS') == '1', 'opt-in CUDA physics')
class ImageRuntime(unittest.TestCase):
    def test_mixed_mesh_contact_matches_cpu_and_rebinds(self):
        from types import SimpleNamespace
        import mujoco, mujoco_warp as mjw, warp as wp,numpy as np
        from dexonomy.sim.mujoco_warp_mixed import MixedModelBank
        from dexonomy.sim.mujoco_warp_env import check_warp_allocation
        wp.init();wp.set_device('cuda:0')
        hosts=[]
        for i in range(2):
         verts=' '.join(str(v) for xyz in [(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1),(1,1,1)] for v in np.array(xyz)*(.03+.01*i))
         extra='<geom type="mesh" mesh="obj" pos=".08 0 0"/>' if i else ''
         xml=f'<mujoco><option timestep="0.002" tolerance="1e-8"/><asset><mesh name="obj" vertex="{verts}"/></asset><worldbody><geom type="plane" size="1 1 .1"/><body name="object" pos="0 0 .1"><freejoint/><geom type="mesh" mesh="obj"/>{extra}</body></worldbody></mujoco>'
         spec=mujoco.MjSpec.from_string(xml);m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
         hosts.append(SimpleNamespace(_model=m,_data=d,_spec=spec,_obj_active_bodyid=1,_cfg=SimpleNamespace(hand_prefix='hand_')))
        bank=MixedModelBank(hosts,device='cuda:0');p=bank.bind([0,1]);d=mjw.make_data(p.carrier_model,nworld=2,nconmax=32,njmax=128);p.initialize_data(d)
        d.qpos.assign(np.stack([h._data.qpos for h in hosts]).astype(np.float32))
        for i in range(200):mjw.step(p.warp_model,d)
        wp.synchronize()
        for h in hosts:
         for _ in range(200):mujoco.mj_step(h._model,h._data)
        np.testing.assert_allclose(d.qpos.numpy(),np.stack([h._data.qpos for h in hosts]),atol=2e-3)
        check_warp_allocation(d,mjw.OverflowType)
        print('PASS mixed-world CPU/GPU contact comparison',flush=True)
        for row in [0,1]:
         p=bank.bind([row]);d=mjw.make_data(p.carrier_model,nworld=1,nconmax=32,njmax=128);p.initialize_data(d)
         d.qpos.assign(hosts[row]._model.qpos0[None].astype(np.float32))
         for _ in range(200):mjw.step(p.warp_model,d)
         wp.synchronize();check_warp_allocation(d,mjw.OverflowType)
         assert np.isfinite(d.qpos.numpy()).all()
         print('PASS singleton world',row,flush=True)

    def test_nonzero_noslip_is_not_silently_disabled(self):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from dexonomy.sim.mujoco_warp_env import preserve_model_options
        wp.init()
        model = mujoco.MjModel.from_xml_string('<mujoco><option noslip_iterations="2"/><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        with wp.ScopedDevice('cuda:0'):
            try:
                gpu = mjw.put_model(model)
            except NotImplementedError as error:
                self.assertIn('noslip', str(error))
                return
            if not hasattr(gpu.opt, 'noslip_iterations'):
                with self.assertRaisesRegex(RuntimeError, 'noslip_iterations'):
                    preserve_model_options(model, gpu)
            else:
                preserve_model_options(model, gpu)
