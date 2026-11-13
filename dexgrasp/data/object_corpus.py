"""Reference reader adapter for raw object-only corpus performance comparisons."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import contained


class ObjectCorpusDataset:
    def __init__(self, root, split, scale, *, condition_keys):
        self.root=Path(root).resolve();self.scale=scale;self.keys=tuple(condition_keys)
        manifest=json.loads((self.root/'raw-index.json').read_text())
        if manifest['schema']!='dexgrasp-raw-index-v1' or manifest['coord_scale_m']!=scale:raise ValueError('raw index mismatch')
        self.shards=manifest['shards'];self.ends=np.cumsum([s['count'] for s in self.shards]);self.count=manifest['splits'][split]['count']
        self.split_path=contained(root,manifest['splits'][split]['path']);self._split=None;self._readers={}

    def __len__(self):return self.count

    def __getitems__(self,indices):
        if self._split is None:self._split=np.load(self.split_path,mmap_mode='r',allow_pickle=False)
        ids=np.asarray(indices,dtype=np.int64)
        if (ids<0).any() or (ids>=len(self)).any():raise IndexError('raw row outside split')
        rows=self._split[ids];groups=np.searchsorted(self.ends,rows,side='right')
        result=[None]*len(ids)
        for sid in np.unique(groups):
            locations=np.flatnonzero(groups==sid);entry=self.shards[sid]
            package=Path(entry['package'])
            if str(package) not in self._readers:
                spec=importlib.util.spec_from_file_location('source_reader',package/'reader.py')
                reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
                self._readers[str(package)]=reader
            start=0 if sid==0 else self.ends[sid-1]
            batch=self._readers[str(package)].load_batch(package,entry['shard'],rows[locations]-start)
            if not np.all(batch['coord_scale_m']==self.scale):raise ValueError('raw normalization scale mismatch')
            for j,loc in enumerate(locations):
                result[loc]=dict(condition={k:torch.from_numpy(batch[k][j]) for k in self.keys},keypoints_gt=torch.from_numpy(batch['keypoints_gt'][j]))
        return result

    def __getitem__(self,index):return self.__getitems__([index])[0]
    def __getstate__(self):return {**self.__dict__,'_split':None,'_readers':{}}
    def close(self):self._split=None;self._readers.clear()
