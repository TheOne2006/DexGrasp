"""Surface sampling and palm rays in metres, independent of a renderer."""
from pathlib import Path
import numpy as np


def triangles(vertices, faces):
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('invalid mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or not len(faces):
        raise ValueError('triangular integer faces required')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('face indices out of bounds')
    return vertices[faces]


def sample_surface(vertices, faces, count, rng):
    tri = triangles(vertices, faces)
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1) * 0.5
    if area.sum() <= 0 or count < 1:
        raise ValueError('empty-area mesh or invalid sample count')
    ids = rng.choice(len(tri), count, p=area/area.sum())
    uv = rng.random((count, 2))
    r, v = np.sqrt(uv[:, 0:1]), uv[:, 1:2]
    selected = tri[ids]
    return ((1-r)*selected[:, 0] + r*(1-v)*selected[:, 1] + r*v*selected[:, 2]).astype(np.float32)


def sample_observation(target, environment, rng):
    obj = sample_surface(*target, 1024, rng)
    scene_obj = sample_surface(*target, 512, rng)
    env = sample_surface(*environment, 1536, rng)
    scene = np.concatenate((scene_obj, env))
    mask = np.concatenate((np.ones(512, np.uint8), np.zeros(1536, np.uint8)))
    order = rng.permutation(2048)
    return dict(object_points=obj, scene_points=scene[order], scene_mask=mask[order])


def combine_meshes(meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        triangles(v, f)
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not vertices:
        raise ValueError('scene requires environment/table surface geometry')
    return np.concatenate(vertices), np.concatenate(faces)


def load_mesh(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            mesh = data['vertices'].copy(), data['faces'].copy()
    else:
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError('OBJ/STL/PLY import needs optional trimesh; NPZ vertices/faces works without it') from exc
        mesh_obj = trimesh.load(path, force='mesh', process=False)
        mesh = np.asarray(mesh_obj.vertices), np.asarray(mesh_obj.faces)
    triangles(*mesh)
    return mesh


def first_ray_hits(origins, direction, vertices, faces, chunk_size=2048):
    """Two-sided Möller–Trumbore; minimum nonnegative distance, inf for misses."""
    tri = triangles(vertices, faces)
    origins = np.asarray(origins, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3 or not np.isfinite(origins).all():
        raise ValueError('invalid ray origins')
    if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1, atol=1e-6):
        raise ValueError('ray direction must be unit length')
    hits = np.full(len(origins), np.inf)
    for start in range(0, len(tri), chunk_size):
        a, b, c = np.moveaxis(tri[start:start+chunk_size], 1, 0)
        e1, e2 = b-a, c-a
        p = np.cross(direction, e2)
        determinant = np.einsum('ij,ij->i', e1, p)
        valid = np.abs(determinant) > 1e-12
        inv = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
        delta = origins[:, None] - a[None]
        u = np.einsum('rfi,fi->rf', delta, p) * inv
        cross = np.cross(delta, e1)
        v = np.einsum('rfi,i->rf', cross, direction) * inv
        distance = np.einsum('rfi,fi->rf', cross, e2) * inv
        ok = valid & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance >= -1e-10)
        hits = np.minimum(hits, np.where(ok, np.maximum(distance, 0), np.inf).min(-1))
    return hits


def inside_closed_mesh(points, vertices, faces, chunk_size=2048):
    """Solid-angle winding number, requiring a closed consistently wound target mesh."""
    tri = triangles(vertices, faces)
    angles = np.zeros(len(points))
    for start in range(0, len(tri), chunk_size):
        v = tri[None, start:start+chunk_size] - np.asarray(points)[:, None, None]
        a, b, c = v[:, :, 0], v[:, :, 1], v[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum('rfi,rfi->rf', a, np.cross(b, c))
        denom = la*lb*lc + np.sum(a*b, -1)*lc + np.sum(b*c, -1)*la + np.sum(c*a, -1)*lb
        angles += (2 * np.arctan2(det, denom)).sum(-1)
    return np.abs(angles) > 2*np.pi


def validate_closed_mesh(vertices, faces):
    triangles(vertices, faces)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    if (counts != 2).any() or (orientation != 0).any():
        raise ValueError('palm labelling requires a closed consistently wound target mesh')


def measure_palm_gap(vertices, faces, rotation, translation, palm, *, check_mesh=True):
    origins = np.asarray(palm['ray_origins_m'], dtype=np.float64)
    normal = np.asarray(palm['normal'], dtype=np.float64)
    if len(origins) < 64:
        raise ValueError('v1 requires at least 64 palm rays')
    if check_mesh:
        validate_closed_mesh(vertices, faces)
    world_origins = origins @ np.asarray(rotation).T + translation
    if inside_closed_mesh(world_origins, vertices, faces).any():
        return dict(valid=False, reason='palm_origin_inside_target', h=None, hit_ratio=0.0, hit_count=0)
    hits = first_ray_hits(world_origins, np.asarray(rotation) @ normal, vertices, faces)
    count = int(np.isfinite(hits).sum())
    valid = count >= 64
    return dict(valid=valid, reason=None if valid else 'fewer_than_64_hits',
                h=float(np.median(hits[np.isfinite(hits)])) if valid else None,
                hit_ratio=count/len(origins), hit_count=count)
