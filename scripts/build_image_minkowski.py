"""Build the real MinkowskiEngine for the /opt/isaacsim Blackwell image.

Downloads checksum-verified NVIDIA source/toolkit into ignored .tools; does not
install a system toolkit, alter drivers, or replace the sparse backbone.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'scripts/patches'
CACHE = ROOT / '.tools/image-build'
CACHE.mkdir(parents=True, exist_ok=True)


def fetch(url, target, digest):
    if not target.exists() or hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != digest:
        temporary = target.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            import shutil
            shutil.copyfileobj(response, output)
        if hashlib.file_digest(temporary.open('rb'), 'sha256').hexdigest() != digest:
            raise ValueError('Download digest mismatch: ' + url)
        temporary.replace(target)
    return target


def extract(archive, destination):
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            member.name = str(Path(*parts[1:]))
            source.extract(member, destination, filter='data')


def main():
    import torch
    if torch.version.cuda != '12.8':
        raise RuntimeError('This profile requires image PyTorch CUDA 12.8')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scipy-openblas32==0.3.34.481.3', 'wheel'], check=True)
    cuda = CACHE / 'cuda-12.8'
    manifest = json.loads((PATCHES/'cuda-12.8.1.json').read_text())
    for name, package in manifest.items():
        entry = package['linux-x86_64']
        archive = fetch('https://developer.download.nvidia.com/compute/cuda/redist/'+entry['relative_path'], CACHE/(name+'.tar.xz'), entry['sha256'])
        extract(archive, cuda)
    (cuda/'lib64').symlink_to('lib', target_is_directory=True) if not (cuda/'lib64').exists() else None
    import scipy_openblas32
    site = Path(scipy_openblas32.__file__).parent.parent
    for package, library in [('cusparse', 'libcusparse.so'), ('cublas', 'libcublas.so')]:
        target = next((site/'nvidia'/package/'lib').glob(library+'.*'))
        link = cuda/'lib'/library
        if not link.exists():
            link.symlink_to(target)
    spec = json.loads((PATCHES/'minkowski-source.json').read_text())
    archive = fetch(spec['url'], CACHE/'minkowski.tar.gz', spec['sha256'])
    source = CACHE/'MinkowskiEngine'
    if not source.exists():
        extract(archive, source)
        subprocess.run(['patch', '-p1', '-i', str(PATCHES/'minkowski-isaac-cu128.patch')], cwd=source, check=True)
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='2', OMP_NUM_THREADS='2')
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--force_cuda', '--blas=openblas', '--blas_include_dirs='+str(site/'scipy_openblas32/include')], cwd=source, env=env, check=True)
    wheel = next((source/'dist').glob('*.whl'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)], check=True)


if __name__ == '__main__':
    main()
