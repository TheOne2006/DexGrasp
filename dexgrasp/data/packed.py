"""Read-only, memory-mapped training tables with shared observation rows.

The format stores deterministic input tensors, never learned features. It is
independent of source package names, sample counts and sampling order. OS paging
bounds resident memory; new source adapters can publish the same table contract.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import contained


class PackedTrainingDataset:
    def __init__(self, root, split, scale, *, condition_keys):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'packed.json').read_text())
        if self.manifest['schema']!='dexgrasp-packed-v1' or self.manifest['coord_scale_m']!=scale:
            raise ValueError('packed schema or scale mismatch')
        self.keys=tuple(condition_keys)
        if set(self.keys)-set(self.manifest['condition']):raise ValueError('packed conditions unavailable')
        self.split=split;self._arrays={}
        self.count=self.manifest['splits'][split]['count']

    def __len__(self):return self.count

    def array(self,name):
        if name not in self._arrays:
            spec=self.manifest['arrays'][name]
            a=np.load(contained(self.root,spec['path']),mmap_mode='r',allow_pickle=False)
            if list(a.shape)!=spec['shape'] or a.dtype.str!=spec['dtype']:raise ValueError('packed array identity mismatch')
            self._arrays[name]=a
        return self._arrays[name]

    def __getitems__(self,indices):
        ids=np.asarray(indices,dtype=np.int64)
        if ids.ndim!=1 or (ids<0).any() or (ids>=len(self)).any():raise IndexError('packed row outside split')
        rows=self.array(self.manifest['splits'][self.split]['array'])[ids]
        joins={}
        def gather(spec):
            selection=rows
            if spec.get('index'):
                index=spec['index']
                if index not in joins:joins[index]=self.array(index)[rows]
                selection=joins[index]
            return torch.from_numpy(self.array(spec['array'])[selection])
        return dict(condition={k:gather(self.manifest['condition'][k]) for k in self.keys},
                    keypoints_gt=gather(self.manifest['target']))

    def __getitem__(self,index):
        b=self.__getitems__([index])
        return dict(condition={k:v[0] for k,v in b['condition'].items()},keypoints_gt=b['keypoints_gt'][0])

    def __getstate__(self):return {**self.__dict__,'_arrays':{}}
    def close(self):self._arrays.clear()
