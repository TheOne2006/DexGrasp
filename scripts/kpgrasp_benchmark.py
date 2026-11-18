"""Load the upstream Shadow Final_base model and optimizer in the benchmark process."""
import os
from pathlib import Path
import subprocess
import torch


def build_reference(source, device, rank, world):
    import dexgrasp
    from hydra import compose, initialize_config_dir
    src = Path(source).resolve()
    dexgrasp.__path__.append(str(src/'src/dexgrasp'))
    from dexgrasp.network.model import KeyPointGraspModel
    from dexgrasp.network.muon import MuonWithAuxAdamW
    os.environ['KPGRASP_ROOT'] = str(src)
    with initialize_config_dir(config_dir=str(src/'src/dexgrasp/config'), version_base=None):
        cfg = compose(config_name='base', overrides=['+experiment=Final_base'])
    model = KeyPointGraspModel(cfg.algo.model).to(device)
    hidden, auxiliary = [], []
    for name, value in model.named_parameters():
        skips = ('norm', 'adaln', 'final_layer', 'embed', 'rms', 'bn')
        group = hidden if value.ndim >= 2 and not any(s in name.lower() for s in skips) else auxiliary
        group.append(value)
    lr = cfg.algo.lr
    optimizer = MuonWithAuxAdamW([dict(params=hidden, use_muon=True), dict(params=auxiliary, use_muon=False)],
                                dict(lr=lr, weight_decay=.01, momentum=.95, rank=rank, world_size=world),
                                dict(lr=lr))
    identity = dict(family='KPGrasp', experiment='Final_base', keypoints=28, stages=3, optimizer='upstream Muon',
                    revision=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'], text=True).strip())
    return model, optimizer, identity


def reference_batch(batch, seed, points):
    rng = torch.Generator().manual_seed(seed)
    data = {key: torch.randn(batch, 1, 28, 3, generator=rng).pin_memory()
            for key in ('pregrasp_kp', 'grasp_kp', 'squeeze_kp')}
    data['point_clouds'] = (points * .2).pin_memory()
    return data
