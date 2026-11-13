"""Validate actual GPU world parameters against immutable native CPU models."""
import numpy as np
GEOM=('geom_pos','geom_quat','geom_size','geom_friction')
BODY=('body_mass','body_inertia','body_ipos','body_iquat')

def native_rows(bank,mesh_names):
    result={name:[] for name in GEOM+BODY+('geom_dataid',)}
    for i,host in enumerate(bank.hosts):
        model=host._model;mapping=bank._template.geom_maps[i]
        for name in GEOM:
            original=np.asarray(getattr(model,name),dtype=np.float32)
            value=np.zeros((bank.carrier_model.ngeom,)+original.shape[1:],dtype=np.float32)
            if name=='geom_quat':value[:,0]=1
            value[mapping]=original;result[name].append(value)
        for name in BODY:result[name].append(np.asarray(getattr(model,name),dtype=np.float32).copy())
        ids=np.full(bank.carrier_model.ngeom,-1,dtype=np.int32)
        for old,new in enumerate(mapping):
            mesh=int(model.geom_dataid[old])
            if mesh>=0:ids[new]=bank.carrier_model.mesh(mesh_names[i][model.mesh(mesh).name]).id
        result['geom_dataid'].append(ids)
    return {name:np.stack(rows) for name,rows in result.items()}

def verify_rows(expected,model,indices):
    for name,rows in expected.items():
        actual=getattr(model,name).numpy()
        if len(actual)==1:actual=np.repeat(actual,len(indices),axis=0)
        # Singleton missing-mesh compatibility duplicates only the lookup row.
        if name=='geom_dataid' and len(indices)==1:actual=actual[:1]
        wanted=rows[indices]
        if not np.array_equal(actual,wanted):
            raise ValueError('GPU world-to-native-model parameter mismatch: '+name)
    return len(indices)
