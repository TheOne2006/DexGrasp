"""Metadata first; GPU inspection is called only inside the guarded worker."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

PHYSICS = {'mujoco':'3.17.0', 'mujoco-warp':'3.17.0', 'warp-lang':'1.16.0'}


def version_tuple(value):
    return tuple(int(x) for x in re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', value).groups(default='0'))


def check_device_build(capability, torch_version, cuda_version, architectures, driver):
    """Pure check, independently testable without initializing a GPU."""
    if capability[0] >= 10:
        if version_tuple(torch_version) < (2,7,0) or not cuda_version or version_tuple(cuda_version) < (12,8,0):
            raise ValueError('Blackwell requires the candidate Torch>=2.7 / CUDA>=12.8 environment')
        arch = 'sm_'+str(capability[0])+str(capability[1])
        if arch not in architectures:
            raise ValueError('Torch wheel does not list the selected Blackwell architecture')
        if version_tuple(driver) < (570,86,10):
            raise ValueError('Blackwell candidate requires a hardware-compatible CUDA12.8-era driver')
    if version_tuple(driver)[0] < 525:
        raise ValueError('Warp CUDA12 build requires NVIDIA driver>=525')


def inspect_runtime(*, gpu=False):
    versions = {n: importlib.metadata.version(n) for n in (*PHYSICS, 'torch', 'numpy')}
    for name, expected in PHYSICS.items():
        if versions[name] != expected:
            raise ValueError(f'{name} must remain {expected}; physics upgrades need separate validation')
    result = dict(versions=versions, gpu='NOT_RUN', physical_validation='NOT_RUN')
    if not gpu:
        return result
    if not os.environ.get('DEXONOMY_PID_RECEIPT') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('GPU preflight requires the guarded runner and explicit device binding')
    receipt = json.loads(Path(os.environ['DEXONOMY_PID_RECEIPT']).read_text())
    if receipt.get('status') != 'PASS' or receipt.get('local_pid') != os.getpid():
        raise ValueError('GPU preflight PID challenge does not belong to this process')
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('one visible GPU required')
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=5).strip()
    capability = (props.major, props.minor)
    architectures = torch.cuda.get_arch_list()
    check_device_build(capability, versions['torch'], torch.version.cuda, architectures, driver)
    import warp as wp
    device = wp.get_device('cuda:0')
    if not device.is_cuda:
        raise ValueError('Warp CUDA device unavailable')
    result.update(gpu=dict(name=props.name, compute_capability=capability,
        total_memory_bytes=props.total_memory, driver=driver, torch_cuda=torch.version.cuda,
        torch_architectures=architectures, warp_device=str(device)),
        environment_profile='rtx4090-tested' if '4090' in props.name and versions['torch']=='2.5.1+cu121'
            else 'candidate-NOT_VALIDATED')
    return result


def main():
    argparse.ArgumentParser(description='CPU-only dependency metadata check; GPU preflight is run by the guard.').parse_args()
    print(json.dumps(inspect_runtime(), indent=2))


if __name__ == '__main__':
    main()
