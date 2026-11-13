"""Small Warp kernels for table contacts and segment-wise external force."""
import warp as wp
wp.set_module_options({'enable_backward':False})

@wp.kernel
def apply_force(body:wp.array[int], forces:wp.array[wp.vec3], com:wp.array2d[wp.vec3],
                xipos:wp.array2d[wp.vec3], xfrc:wp.array2d[wp.spatial_vector]):
    w=wp.tid();b=body[w];f=forces[w]
    xfrc[w,b]=wp.spatial_vector(f,wp.cross(com[w,b]-xipos[w,b],f))

@wp.kernel
def table_contacts(nacon:wp.array[int], worldid:wp.array[int], geom:wp.array[wp.vec2i],
                   dist:wp.array[float], categories:wp.array2d[int], tolerance:float,
                   counts:wp.array2d[int], near:wp.array2d[int], minimum:wp.array2d[float], invalid:wp.array[int]):
    i=wp.tid()
    if i<nacon[0]:
        w=worldid[i];a=geom[i][0];b=geom[i][1]
        if w<0 or w>=categories.shape[0] or a<0 or b<0 or a>=categories.shape[1] or b>=categories.shape[1]:
            wp.atomic_add(invalid,0,1)
        else:
            ca=categories[w,a];cb=categories[w,b];other=int(0)
            if ca==1:other=cb
            elif cb==1:other=ca
            if other==2 or other==3:
                j=other-2;value=dist[i]
                if not wp.isfinite(value):wp.atomic_add(invalid,0,1)
                else:
                    wp.atomic_min(minimum,w,j,value)
                    if value<=tolerance:wp.atomic_add(counts,w,j,1)
                    else:wp.atomic_add(near,w,j,1)
