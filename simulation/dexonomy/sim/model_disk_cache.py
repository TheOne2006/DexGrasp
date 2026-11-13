"""Content-addressed CPU MJB cache. GPU models and graphs remain process-local."""
import dataclasses,gzip,hashlib,json,os,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from dexonomy.acceptance_audit import event


_HASHES = {}

def file_hash(path):
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if key not in _HASHES:_HASHES[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    return _HASHES[key]

def encode(x):
    if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def cache_key(hand,scene,sim):
    deps={}
    def file(path):
        path=Path(path).resolve()
        if str(path) in deps:return
        deps[str(path)]=file_hash(path)
        if path.suffix.lower()!='.xml':return
        tree=ET.parse(path).getroot();compiler=tree.find('compiler')
        meshdir=compiler.get('meshdir','') if compiler is not None else ''
        texturedir=compiler.get('texturedir','') if compiler is not None else ''
        for node in tree.iter():
            value=node.get('file')
            if not value:continue
            base=meshdir if node.tag=='mesh' else texturedir if node.tag=='texture' else ''
            file(path.parent/base/value)
    def walk(x):
        if dataclasses.is_dataclass(x):walk(dataclasses.asdict(x))
        elif isinstance(x,dict):
            for v in x.values():walk(v)
        elif isinstance(x,(tuple,list)):
            for v in x:walk(v)
        elif isinstance(x,(str,Path)) and Path(x).is_file():file(x)
    walk(hand);walk(scene)
    source_root=Path(__file__).resolve().parent
    sources={p.name:file_hash(p) for p in source_root.glob('*.py')}
    payload={'schema':3,'hand':hand,'scene':scene,'scene_order':list(scene.get('scene',{})) if scene else [],'sim':sim,'dependencies':deps,'sources':sources,'mujoco':mujoco.__version__}
    raw=json.dumps(payload,default=encode,sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest(),json.loads(raw)


def compile_cached(spec,hand,scene,sim):
    directory=os.environ.get('DEXONOMY_MJB_CACHE')
    if not directory:return spec.compile()
    started=time.perf_counter();key,manifest=cache_key(hand,scene,sim);key_s=time.perf_counter()-started
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);binary=root/(key+'.mjb.gz');metadata=root/(key+'.json')
    hit=False;begin=time.perf_counter()
    if binary.exists() and metadata.exists():
        saved=json.loads(metadata.read_text());compressed=binary.read_bytes()
        if saved['sha256']!=hashlib.sha256(compressed).hexdigest():
            raise ValueError('Corrupted MJB cache: '+str(binary))
        raw=gzip.decompress(compressed)
        if saved['raw_sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('Corrupted decompressed MJB: '+str(binary))
        model=mujoco.MjModel.from_binary_path('model.mjb',{'model.mjb':raw});hit=True
    if not hit:
        model=spec.compile();buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
        mujoco.mj_saveModel(model,None,buffer);raw=buffer.tobytes();compressed=gzip.compress(raw,compresslevel=1,mtime=0)
        temp=root/(key+f'.{os.getpid()}.partial');temp.write_bytes(compressed);os.replace(temp,binary)
        tempmeta=root/(key+f'.{os.getpid()}.json.partial')
        tempmeta.write_text(json.dumps({'key':key,'sha256':hashlib.sha256(compressed).hexdigest(),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'stored_bytes':len(compressed),'manifest':manifest},indent=2));os.replace(tempmeta,metadata)
    event('cpu_mjb_cache',key=key,hit=hit,key_s=key_s,raw_bytes=len(raw),stored_bytes=len(compressed),compile_or_read_s=time.perf_counter()-begin,wall_s=time.perf_counter()-started,nq=model.nq,nv=model.nv,ngeom=model.ngeom,nmesh=model.nmesh)
    return model
