"""Per-family active FIFO; complete immutable feedback is held by the stage ledger."""
import collections,random
from pathlib import Path
import numpy as np
import torch
from .hand_loader import MultiHandTemplateLoader

class FamilyPoolLoader(MultiHandTemplateLoader):
    def __init__(self,*,family_capacity=512,**kwargs):
        if not 1<=family_capacity<=4096:raise ValueError('family_capacity must be 1..4096 evolved templates')
        self.family_capacity=int(family_capacity)
        kwargs['max_data_buffer']=(self.family_capacity+1)*len(kwargs['tmpl_names'])
        super().__init__(**kwargs)

    def _append_hand_data(self,hand_data):
        family=hand_data['tmpl_name']
        if not self._accepts_template_name(family):return False
        with self.buffer_lock:
            for key,value in self.family_extra_info[family].items():
                if hand_data[key]!=value:raise ValueError('family metadata mismatch: '+key)
            if hand_data['hand_path'] in self.data_buffer['hand_path']:return False
            seed_count=len(self.template_names)
            active=[i for i,x in enumerate(self.data_buffer['tmpl_name']) if i>=seed_count and x==family]
            if len(active)>=self.family_capacity:
                for values in self.data_buffer.values():values.pop(active[0])
            for key,values in self.data_buffer.items():
                value=hand_data[key];values.append(torch.from_numpy(value) if isinstance(value,np.ndarray) else value)
            size=len(self.data_buffer['hand_path'])
        with self.feedback_stats_lock:self.feedback_stats['buffer_peak']=max(size,self.feedback_stats['buffer_peak'])
        return True

    def activate_family(self,family,references,root,seed):
        """Simple random subset of the entire committed family history, including evictions.

        References point to immutable feedback payloads; no historical candidate is
        deleted. Activation does not append feedback IDs or increment acceptance.
        """
        if family not in self.template_names:raise ValueError('unknown family')
        chosen=random.Random(seed).sample(references,min(len(references),self.family_capacity))
        bundles={};prepared=[]
        for ref in chosen:
            path=Path(root)/ref['path']
            if path not in bundles:bundles[path]=np.load(path,allow_pickle=True).item()
            item=bundles[path]['prepared'][ref['index']]
            assert item['hand_path']==ref['id'] and item['tmpl_name']==family
            prepared.append(item)
        with self.buffer_lock:
            keep=[i for i,f in enumerate(self.data_buffer['tmpl_name']) if i<len(self.template_names) or f!=family]
            self.data_buffer={k:[v[i] for i in keep] for k,v in self.data_buffer.items()}
        for item in prepared:assert self._append_hand_data(item)
        return [x['hand_path'] for x in prepared]

    def distribution(self):
        with self.buffer_lock:return dict(collections.Counter(self.data_buffer['tmpl_name']))

    def memory_estimate(self):
        with self.buffer_lock:
            sizes=[sum(v[i].numel()*v[i].element_size() if isinstance(v[i],torch.Tensor) else 256 for v in self.data_buffer.values()) for i in range(len(self.data_buffer['hand_path']))]
        return {'sample_max_payload_bytes':max(sizes),'configured_payload_upper_estimate_bytes':max(sizes)*(self.family_capacity+1)*len(self.template_names),'note':'Tensor payload plus conservative metadata allowance; Python allocator/process models require separate measured headroom.'}
