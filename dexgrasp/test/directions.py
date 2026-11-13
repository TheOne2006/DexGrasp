"""Controlled direction sets and metrics; axes are expressed in the scene frame."""
from itertools import product
import numpy as np


def cube_directions():
    """26 unique directions: 6 face normals, 12 edge diagonals, 8 corner diagonals."""
    result=[]
    for group, count in (('axis',1),('face_diagonal',2),('space_diagonal',3)):
        for xyz in product((-1,0,1),repeat=3):
            if sum(v!=0 for v in xyz)!=count:
                continue
            vector=np.asarray(xyz,dtype=float);vector/=np.linalg.norm(vector)
            label=' '.join(('+' if value>0 else '−')+axis for axis,value in zip('XYZ',xyz) if value)
            ident=''.join(axis+('p' if value>0 else 'm') for axis,value in zip('xyz',xyz) if value)
            result.append(dict(direction_id=ident,direction_label=label,direction_group=group,command_direction=vector.tolist()))
    return result


def direction_summary(rows, split, tolerance_deg=10.):
    subset=[r for r in rows if split=='all' or r['split']==split]
    def mean(key):
        values=[r[key] for r in subset if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else None
    n=len(subset)
    return dict(split=split,samples=n,objects=len({r['object_id'] for r in subset}),
                decoded=sum(r.get('R') is not None for r in subset),
                direction_error_deg=mean('direction_error_deg'),ik_rmse_mm=mean('ik_rmse_mm'),
                direction_pass_rate=sum(r.get('direction_error_deg') is not None and r['direction_error_deg']<=tolerance_deg for r in subset)/n if n else None,
                ik_success_rate=sum(r.get('ik_success') is True for r in subset)/n if n else None,
                palm_gap_coverage=sum(r.get('palm_gap_valid') is True for r in subset)/n if n else None,
                palm_gap_error_mm=mean('palm_gap_error_mm'),
                geometric_pass_rate=sum(r.get('geometric_pass') is True for r in subset)/n if n else None,
                physical_success_rate=None)
