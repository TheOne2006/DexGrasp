"""Portable packed table semantics; no optimizer updates."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dexgrasp.data.packed import PackedTrainingDataset
from dexgrasp.train.distributed import training_collate


class PackedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        values={'points':np.arange(24,dtype=np.float32).reshape(2,4,3),'obs':np.array([1,0,1]),
                'K':np.arange(27,dtype=np.float32).reshape(3,3,3),'d':np.eye(3,dtype=np.float32),
                'train':np.array([2,0,1]),'test':np.array([],dtype=np.int64)}
        arrays={}
        for k,a in values.items():
            np.save(self.root/(k+'.npy'),a);arrays[k]=dict(path=k+'.npy',shape=list(a.shape),dtype=a.dtype.str)
        self.values=values
        self.manifest=dict(schema='dexgrasp-packed-v1',coord_scale_m=.2,arrays=arrays,
           splits={'train':dict(array='train',count=3),'test':dict(array='test',count=0)},
           condition={'object_points':dict(array='points',index='obs'),'grasp_direction_cmd':dict(array='d')},target=dict(array='K'))
        (self.root/'packed.json').write_text(json.dumps(self.manifest))
    def tearDown(self):self.temp.cleanup()
    def dataset(self):return PackedTrainingDataset(self.root,'train',.2,condition_keys=('object_points','grasp_direction_cmd'))
    def test_batch_gather_preserves_split_order_duplicates_and_join(self):
        d=self.dataset();b=d.__getitems__([2,0,2,1]);rows=np.array([1,2,1,0])
        np.testing.assert_array_equal(b['keypoints_gt'].numpy(),self.values['K'][rows])
        np.testing.assert_array_equal(b['condition']['object_points'].numpy(),self.values['points'][self.values['obs'][rows]])
        self.assertIs(training_collate(b),b)
        np.testing.assert_array_equal(d[0]['keypoints_gt'].numpy(),self.values['K'][2])
    def test_spawn_reopens_maps_and_returns_writable_independent_batches(self):
        d=self.dataset();first=d.__getitems__([0]);first['keypoints_gt'].zero_()
        child=pickle.loads(pickle.dumps(d));self.assertFalse(child._arrays)
        np.testing.assert_array_equal(child[0]['keypoints_gt'].numpy(),self.values['K'][2])
        d.close();self.assertFalse(d._arrays)
    def test_invalid_rows_scale_shapes_and_paths_fail(self):
        d=self.dataset()
        for ids in ([-1],[3]):
            with self.assertRaises(IndexError):d.__getitems__(ids)
        with self.assertRaises(ValueError):PackedTrainingDataset(self.root,'train',.3,condition_keys=('object_points',))
        with self.assertRaises(ValueError):PackedTrainingDataset(self.root,'train',.2,condition_keys=('scene_points',))
        d.manifest['arrays']['points']['shape']=[100]
        with self.assertRaises(ValueError):d.__getitems__([0])
        d=self.dataset();d.manifest['arrays']['points']['path']='../external.npy'
        with self.assertRaises(ValueError):d.__getitems__([0])


if __name__=='__main__':unittest.main()
