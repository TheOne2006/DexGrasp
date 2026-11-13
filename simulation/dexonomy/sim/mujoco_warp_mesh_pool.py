"""Restore native compiled mesh coordinates and convex topology in a pool.

The pool compiler may need convex inertia to allocate hull buffers for meshes
that are referenced only by another world's geom_dataid. That compiler choice
must not change the native mesh coordinate frame or collision features. Only
global buffer addresses are relocated here; vertex and polygon IDs remain local
to their mesh, as required by MuJoCo Warp's GJK and MULTICCD implementations.
"""
from __future__ import annotations

import numpy as np


def _span(model, mesh_id, address, count):
    start, length = int(getattr(model, address)[mesh_id]), int(getattr(model, count)[mesh_id])
    if length < 0 or (length and start < 0):
        raise ValueError(f"invalid compiled mesh span: {address}/{count}")
    return slice(max(0, start), max(0, start) + length), length


def _graph_span(model, mesh_id):
    start = int(model.mesh_graphadr[mesh_id])
    if start < 0:
        return slice(0, 0), 0
    following = model.mesh_graphadr[model.mesh_graphadr > start]
    end = int(np.min(following)) if len(following) else len(model.mesh_graph)
    return slice(start, end), end - start


def _ragged_plan(source, target, source_rows, target_rows, address, count, payload):
    source_counts = np.array(getattr(source, count)[source_rows], copy=True)
    target_counts = np.asarray(getattr(target, count)[target_rows])
    source_addresses = np.array(getattr(source, address)[source_rows], copy=True)
    target_addresses = np.asarray(getattr(target, address)[target_rows])
    if source_counts.shape != target_counts.shape or int(source_counts.sum()) != int(target_counts.sum()):
        raise ValueError(f"compiled mesh pool buffer length differs: {payload}")
    length = int(source_counts.sum())
    writes = [(getattr(target, count), target_rows, source_counts)]
    if not length:
        # Empty rows are never dereferenced; preserve the native sentinel.
        writes.append((getattr(target, address), target_rows, source_addresses))
        return writes
    source_nonempty, target_nonempty = source_counts > 0, target_counts > 0
    source_start = int(source_addresses[source_nonempty].min())
    target_start = int(target_addresses[target_nonempty].min())
    for addresses, counts, nonempty, start in (
        (source_addresses, source_counts, source_nonempty, source_start),
        (target_addresses, target_counts, target_nonempty, target_start),
    ):
        if start < 0 or int((addresses[nonempty] + counts[nonempty]).max()) != start + length:
            raise ValueError(f"compiled mesh pool ragged data is not contiguous: {payload}")
    delta = target_start - source_start
    relocated = np.where(source_addresses >= 0, source_addresses + delta, source_addresses)
    values = np.array(getattr(source, payload)[source_start:source_start + length], copy=True)
    writes.extend(((getattr(target, address), target_rows, relocated),
                   (getattr(target, payload), slice(target_start, target_start + length), values)))
    return writes


def copy_native_mesh(source, src_id, target, dst_id):
    """Copy one native mesh into already allocated pool buffers, or reject it.

The native and pooled mesh must have identical buffer capacities. Global graph,
vertex, face and polygon base addresses stay those of the pool. Global offsets
inside polygon-to-vertex and vertex-to-polygon tables are shifted accordingly;
their payloads contain mesh-local IDs and are copied without an ID offset.
This routine covers rigid mesh geometry and convex collision data used by Warp;
it does not support mesh SDF octrees or update MuJoCo's unused CPU mesh BVH.
All size checks run before any target array is changed.
"""
    src_id, dst_id = int(src_id), int(dst_id)
    if not 0 <= src_id < source.nmesh or not 0 <= dst_id < target.nmesh:
        raise ValueError("mesh ID outside compiled mesh pool")
    if int(source.mesh_octnum[src_id]) or int(target.mesh_octnum[dst_id]):
        raise ValueError("native mesh pool copying does not support SDF octrees")
    writes = []
    ranges = {}
    for kind, address, count, fields in (
        ("vertex", "mesh_vertadr", "mesh_vertnum", ("mesh_vert",)),
        ("normal", "mesh_normaladr", "mesh_normalnum", ("mesh_normal",)),
        ("face", "mesh_faceadr", "mesh_facenum", ("mesh_face", "mesh_facenormal", "mesh_facetexcoord")),
        ("texcoord", "mesh_texcoordadr", "mesh_texcoordnum", ("mesh_texcoord",)),
        ("polygon", "mesh_polyadr", "mesh_polynum", ("mesh_polynormal",)),
    ):
        src, src_count = _span(source, src_id, address, count)
        dst, dst_count = _span(target, dst_id, address, count)
        if src_count != dst_count:
            raise ValueError(f"native mesh pool {kind} count differs: {src_count} != {dst_count}")
        ranges[kind] = (src, dst)
        for name in fields:
            writes.append((getattr(target, name), dst, np.array(getattr(source, name)[src], copy=True)))
    src_graph, src_length = _graph_span(source, src_id)
    dst_graph, dst_length = _graph_span(target, dst_id)
    if src_length != dst_length:
        raise ValueError(f"native mesh pool convex graph length differs: {src_length} != {dst_length}")
    writes.append((target.mesh_graph, dst_graph, np.array(source.mesh_graph[src_graph], copy=True)))
    for name in ("mesh_pos", "mesh_quat", "mesh_scale", "mesh_extrema"):
        writes.append((getattr(target, name), dst_id, np.array(getattr(source, name)[src_id], copy=True)))
    src_poly, dst_poly = ranges["polygon"]
    writes.extend(_ragged_plan(source, target, src_poly, dst_poly,
                               "mesh_polyvertadr", "mesh_polyvertnum", "mesh_polyvert"))
    src_vert, dst_vert = ranges["vertex"]
    writes.extend(_ragged_plan(source, target, src_vert, dst_vert,
                               "mesh_polymapadr", "mesh_polymapnum", "mesh_polymap"))
    for array, indices, values in writes:
        array[indices] = values


def native_mesh_identity(model, mid):
    """Exact content identity of every copied native mesh field, sans base addresses.

    Never deduplicate by name, scale or path alone. This includes the GJK graph
    and MULTICCD polygon adjacency; geometry/inertia compilation stays unchanged.
    """
    import hashlib
    digest = hashlib.sha256()
    def add(name, value):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    if int(model.mesh_octnum[mid]):
        raise ValueError('native mesh pool copying does not support SDF octrees')
    for address, count, fields in (
        ('mesh_vertadr', 'mesh_vertnum', ('mesh_vert',)),
        ('mesh_normaladr', 'mesh_normalnum', ('mesh_normal',)),
        ('mesh_faceadr', 'mesh_facenum', ('mesh_face', 'mesh_facenormal', 'mesh_facetexcoord')),
        ('mesh_texcoordadr', 'mesh_texcoordnum', ('mesh_texcoord',)),
        ('mesh_polyadr', 'mesh_polynum', ('mesh_polynormal',)),
    ):
        rows, _ = _span(model, mid, address, count)
        for field in fields:
            add(field, getattr(model, field)[rows])
    rows, _ = _graph_span(model, mid)
    add('mesh_graph', model.mesh_graph[rows])
    for field in ('mesh_pos', 'mesh_quat', 'mesh_scale', 'mesh_extrema'):
        add(field, getattr(model, field)[mid])
    for address, count, row_address, row_count, payload in (
        ('mesh_polyvertadr','mesh_polyvertnum','mesh_polyadr','mesh_polynum','mesh_polyvert'),
        ('mesh_polymapadr','mesh_polymapnum','mesh_vertadr','mesh_vertnum','mesh_polymap'),
    ):
        rows, _ = _span(model, mid, row_address, row_count)
        counts = getattr(model,count)[rows]
        addresses = getattr(model,address)[rows]
        add(count, counts)
        nonempty = counts > 0
        if np.any(nonempty):
            start = int(addresses[nonempty].min())
            length = int(counts.sum())
            add(address, np.where(addresses >= 0, addresses-start, addresses))
            add(payload, getattr(model,payload)[start:start+length])
        else:
            add(address, addresses)
    return digest.digest()
