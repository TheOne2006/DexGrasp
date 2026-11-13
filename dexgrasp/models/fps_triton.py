"""Fused deterministic FPS for fixed-size CUDA point clouds.

One program per cloud keeps distances in registers throughout sampling. The
centroid is computed by Torch so tie-breaking and reduction match the reference.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fps(points, centroid, output, N: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < N
    x = tl.load(points + row*N*3 + index*3, valid, 0)
    y = tl.load(points + row*N*3 + index*3+1, valid, 0)
    z = tl.load(points + row*N*3 + index*3+2, valid, 0)
    dx = x-tl.load(centroid + row*3)
    dy = y-tl.load(centroid + row*3+1)
    dz = z-tl.load(centroid + row*3+2)
    radius = tl.where(valid, dx*dx+dy*dy+dz*dz, -1.)
    maximum = tl.max(radius, 0)
    farthest = tl.min(tl.where(valid & (radius == maximum), index, 2147483647), 0)
    distance = tl.full((BLOCK,), float('inf'), tl.float32)
    selected = tl.full((BLOCK,), False, tl.int1)
    for step in range(COUNT):
        tl.store(output + row*COUNT+step, farthest)
        selected = selected | (index == farthest)
        cx = tl.load(points + row*N*3+farthest*3)
        cy = tl.load(points + row*N*3+farthest*3+1)
        cz = tl.load(points + row*N*3+farthest*3+2)
        dx, dy, dz = x-cx, y-cy, z-cz
        distance = tl.minimum(distance, dx*dx+dy*dy+dz*dz)
        candidate = tl.where(valid & ~selected, distance, -1.)
        maximum = tl.max(candidate, 0)
        farthest = tl.min(tl.where(valid & (candidate == maximum), index, 2147483647), 0)


@torch.no_grad()
def farthest_point_indices_triton(points, count):
    if not points.is_cuda or points.ndim != 3 or points.shape[-1] != 3 or not 0 < count <= points.shape[1]:
        raise ValueError('Triton FPS requires CUDA [B,N,3] points and 0 < count <= N')
    xyz = points.float().contiguous()
    batch, size, _ = xyz.shape
    centroid = xyz.mean(1)
    output = torch.empty((batch, count), device=points.device, dtype=torch.long)
    with torch.cuda.device(points.device):
        _fps[(batch,)](xyz, centroid, output, size, count, triton.next_power_of_2(size),
                      num_warps=4, enable_fp_fusion=False)
    return output
