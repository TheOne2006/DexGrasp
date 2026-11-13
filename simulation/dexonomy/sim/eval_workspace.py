"""Bounded, protocol-owned Eval storage. Importing this module is CPU-only.

Owners retain an immutable MixedModelBank. Refine must use a different owner.
The caller supplies all shape/parameter/phase inputs in its key; only one live
workspace is retained, and stage transitions explicitly release it.
"""
import os


class EvalWorkspace:
    def __init__(self, protocol):
        self.protocol = protocol
        self.key = None
        self.value = None
        self.hits = 0
        self.builds = 0
        self.releases = 0

    def acquire(self, key, build, reset):
        key = (self.protocol, key)
        if os.environ.get('DEXONOMY_DATA_REUSE') == '1' and self.key == key and self.value is not None:
            self.hits += 1
            try:
                reset(self.value)
            except BaseException:
                self.release()
                raise
            return self.value
        self.release()
        value = build()
        self.builds += 1
        if os.environ.get('DEXONOMY_DATA_REUSE') == '1':
            self.key, self.value = key, value
        return value

    def release(self):
        self.value = None
        self.key = None
        self.releases += 1

    @property
    def stats(self):
        return dict(protocol=self.protocol, workspace_hits=self.hits,
                    workspace_builds=self.builds, workspace_releases=self.releases)


def eval_allocation(model, capacities=None, scale=1):
    from dexonomy.sim.mujoco_warp_env import mixed_world_buffer_sizes
    from dexonomy.night_contracts import constraint_allocation, contact_allocation
    capacities = capacities or {}
    allocation = constraint_allocation(mixed_world_buffer_sizes(model.nv, scale), capacities.get('njmax', 0))
    return contact_allocation(allocation, capacities.get('nconmax', 0), capacities.get('nccdmax', 0))
