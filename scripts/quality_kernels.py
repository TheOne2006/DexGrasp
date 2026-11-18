"""GPU reduction of signed MuJoCo collision contacts; metres in, metres out."""
import warp as wp
wp.set_module_options({'enable_backward':False})

@wp.kernel
def reduce_contacts(nacon:wp.array[int],worldid:wp.array[int],geom:wp.array[wp.vec2i],dist:wp.array[float],cats:wp.array[int],minima:wp.array2d[float],pd:wp.array[float]):
    i=wp.tid()
    if i<nacon[0]:
        w=worldid[i];a=cats[geom[i][0]];b=cats[geom[i][1]];finger=int(-2)
        if a==5: finger=b
        elif b==5:finger=a
        if finger>=-1 and finger<5:
            wp.atomic_max(pd,w,wp.max(0.0,-dist[i]))
            if finger>=0:wp.atomic_min(minima,w,finger,dist[i])
