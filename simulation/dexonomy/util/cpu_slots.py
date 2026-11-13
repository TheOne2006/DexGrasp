"""Cgroup-aware, priority-ordered CPU slots for streaming MuJoCo stages.

The coordinator creates one :class:`PriorityCpuSlots` instance with the same
``spawn`` multiprocessing context used for stage processes.  Spawned stage
processes may then pass/inherit it into their internal ``fork`` pools.  A lease
is held for one complete scene task, keeping model reuse intact while enforcing
a hard, pipeline-wide limit on concurrently running MuJoCo workers.

Priority is non-preemptive: a newly waiting high-priority stage receives the
next free slot, but an already running lower-priority scene is never interrupted.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_STAGE_PRIORITY = ("grasp", "stability")
SNAPSHOT_FORMAT = "dexonomy-priority-cpu-slots/v1"


class CpuSlotError(RuntimeError):
    """Base class for global CPU slot failures."""


class CpuSlotCancelled(CpuSlotError):
    """A waiting lease was cancelled before it acquired a CPU slot."""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _parse_cpuset(raw: str | None) -> int | None:
    """Return the number of CPUs in Linux's ``0-3,8`` cpuset syntax."""

    if not raw:
        return None
    cpus: set[int] = set()
    try:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                return None
            if "-" in item:
                start_raw, end_raw = item.split("-", 1)
                start, end = int(start_raw), int(end_raw)
                if start < 0 or end < start:
                    return None
                cpus.update(range(start, end + 1))
            else:
                cpu = int(item)
                if cpu < 0:
                    return None
                cpus.add(cpu)
    except (TypeError, ValueError):
        return None
    return len(cpus) or None


def _quota_capacity(cgroup_root: Path) -> int | None:
    """Read a conservative integer CPU quota from cgroup v2 or v1."""

    cpu_max = _read_text(cgroup_root / "cpu.max")
    if cpu_max:
        fields = cpu_max.split()
        if len(fields) == 2 and fields[0] != "max":
            try:
                quota, period = int(fields[0]), int(fields[1])
            except ValueError:
                pass
            else:
                if quota > 0 and period > 0:
                    return max(1, math.floor(quota / period))

    # Common cgroup-v1 layouts are either rooted in ``cpu/`` or presented
    # directly in the supplied controller directory.
    for directory in (cgroup_root / "cpu", cgroup_root):
        quota_raw = _read_text(directory / "cpu.cfs_quota_us")
        period_raw = _read_text(directory / "cpu.cfs_period_us")
        if quota_raw is None or period_raw is None:
            continue
        try:
            quota, period = int(quota_raw), int(period_raw)
        except ValueError:
            continue
        if quota > 0 and period > 0:
            return max(1, math.floor(quota / period))
    return None


def detect_available_cpu_slots(
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    *,
    affinity_count: int | None = None,
    fallback_cpu_count: int | None = None,
) -> int:
    """Return the effective integer CPU capacity visible to this process.

    The result is the minimum positive limit from process affinity, cgroup
    cpuset, and cgroup CPU quota.  Fractional quotas are rounded down so the
    returned value is a safe hard worker count.  At least one slot is returned.

    ``affinity_count`` and ``fallback_cpu_count`` are injectable primarily for
    deterministic tests; normal callers should leave both unset.
    """

    root = Path(cgroup_root)
    limits: list[int] = []

    if affinity_count is not None:
        if (
            isinstance(affinity_count, bool)
            or not isinstance(affinity_count, int)
            or affinity_count <= 0
        ):
            raise ValueError("affinity_count must be a positive integer")
        limits.append(affinity_count)
    else:
        try:
            affinity = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
        if affinity:
            limits.append(int(affinity))

    cpuset_count = None
    for relative in (
        "cpuset.cpus.effective",
        "cpuset/cpuset.cpus.effective",
        "cpuset.cpus",
        "cpuset/cpuset.cpus",
    ):
        cpuset_count = _parse_cpuset(_read_text(root / relative))
        if cpuset_count is not None:
            break
    if cpuset_count is not None:
        limits.append(cpuset_count)

    quota_count = _quota_capacity(root)
    if quota_count is not None:
        limits.append(quota_count)

    if not limits:
        cpu_count = fallback_cpu_count
        if cpu_count is None:
            cpu_count = os.cpu_count()
        if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count <= 0:
            cpu_count = 1
        limits.append(cpu_count)
    return max(1, min(limits))


class _CpuSlotLease(AbstractContextManager["_CpuSlotLease"]):
    def __init__(self, slots: "PriorityCpuSlots", stage: str, cancel_event: Any):
        self._slots = slots
        self._stage = stage
        self._cancel_event = cancel_event
        self._acquired = False

    def __enter__(self) -> "_CpuSlotLease":
        self._slots.acquire(self._stage, cancel_event=self._cancel_event)
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._acquired:
            self._acquired = False
            self._slots.release(self._stage)
        return False


class PriorityCpuSlots:
    """A process-shared, non-preemptive CPU slot gate.

    Without ``stage_targets`` the gate uses strict stage priority. With
    targets, waiting stages below their soft target are restored first and any
    otherwise idle share remains borrowable by another stage.
    """

    def __init__(
        self,
        capacity: int,
        *,
        context: multiprocessing.context.BaseContext | None = None,
        stages: Iterable[str] = DEFAULT_STAGE_PRIORITY,
        stage_targets: Mapping[str, int] | None = None,
        cancel_poll_interval_s: float = 0.05,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("CPU slot capacity must be a positive integer")
        stage_tuple = tuple(stages)
        if (
            not stage_tuple
            or len(set(stage_tuple)) != len(stage_tuple)
            or any(not isinstance(stage, str) or not stage for stage in stage_tuple)
        ):
            raise ValueError("CPU slot stages must be unique, non-empty strings")
        if cancel_poll_interval_s <= 0:
            raise ValueError("cancel_poll_interval_s must be positive")

        ctx = context or multiprocessing.get_context("spawn")
        count = len(stage_tuple)
        self._stages = stage_tuple
        self._stage_index = {stage: index for index, stage in enumerate(stage_tuple)}
        self._cancel_poll_interval_s = float(cancel_poll_interval_s)
        self._condition = ctx.Condition(ctx.RLock())
        # All raw values below are read or written only while the condition is held.
        self._capacity = ctx.Value("i", capacity, lock=False)
        self._initial_capacity = capacity
        self._minimum_capacity = ctx.Value("i", capacity, lock=False)
        self._maximum_capacity = ctx.Value("i", capacity, lock=False)
        self._capacity_changes = ctx.Value("q", 0, lock=False)
        self._active_total = ctx.Value("i", 0, lock=False)
        self._peak_active_total = ctx.Value("i", 0, lock=False)
        self._active = ctx.Array("i", count, lock=False)
        self._waiting = ctx.Array("i", count, lock=False)
        self._peak_active = ctx.Array("i", count, lock=False)
        self._acquired = ctx.Array("q", count, lock=False)
        self._released = ctx.Array("q", count, lock=False)
        self._cancelled = ctx.Array("q", count, lock=False)
        self._blocked = ctx.Array("q", count, lock=False)
        self._wait_ns = ctx.Array("q", count, lock=False)
        self._targets_enabled = ctx.Value("b", stage_targets is not None, lock=False)
        self._targets = ctx.Array("i", count, lock=False)
        self._target_changes = ctx.Value("q", 0, lock=False)
        if stage_targets is not None:
            normalized_targets = self._normalize_targets(stage_targets, capacity)
            for index, target in enumerate(normalized_targets):
                self._targets[index] = target

    @property
    def stages(self) -> tuple[str, ...]:
        return self._stages

    def _index(self, stage: str) -> int:
        try:
            return self._stage_index[stage]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"unknown CPU slot stage {stage!r}; expected one of {self._stages}"
            ) from exc

    def _normalize_targets(
        self, stage_targets: Mapping[str, int], capacity: int
    ) -> tuple[int, ...]:
        if not isinstance(stage_targets, Mapping):
            raise ValueError("stage_targets must be a mapping or None")
        unknown = set(stage_targets).difference(self._stages)
        if unknown:
            raise ValueError(f"unknown CPU slot target stages: {sorted(unknown)}")
        normalized = []
        for stage in self._stages:
            target = stage_targets.get(stage, 0)
            if (
                isinstance(target, bool)
                or not isinstance(target, int)
                or target < 0
            ):
                raise ValueError(
                    f"CPU slot target for {stage} must be a non-negative integer"
                )
            normalized.append(target)
        if sum(normalized) > capacity:
            raise ValueError(
                "sum of CPU slot stage targets cannot exceed global capacity "
                f"({sum(normalized)} > {capacity})"
            )
        return tuple(normalized)

    @staticmethod
    def _cancelled_event(cancel_event: Any) -> bool:
        return cancel_event is not None and bool(cancel_event.is_set())

    def acquire(self, stage: str, *, cancel_event: Any = None) -> None:
        """Wait for and acquire one slot for ``stage``.

        Higher-priority stages are those appearing earlier in ``stages``.  A
        lower-priority waiter can enter only when capacity is available and no
        higher-priority process is already waiting.
        """

        index = self._index(stage)
        wait_started_ns = time.monotonic_ns()
        with self._condition:
            self._waiting[index] += 1
            blocked_recorded = False
            try:
                while True:
                    if self._cancelled_event(cancel_event):
                        self._cancelled[index] += 1
                        raise CpuSlotCancelled(
                            f"CPU slot acquisition was cancelled for stage {stage}"
                        )
                    capacity_available = self._active_total.value < self._capacity.value
                    if self._targets_enabled.value:
                        # Restore waiting stages that are below their soft target
                        # before granting borrowed capacity. Existing priority
                        # orders multiple under-target stages. A stage with no
                        # waiter never leaves its target share idle.
                        eligible = [
                            candidate
                            for candidate in range(len(self._stages))
                            if self._waiting[candidate] > 0
                            and self._active[candidate] < self._targets[candidate]
                        ]
                        if not eligible:
                            eligible = [
                                candidate
                                for candidate in range(len(self._stages))
                                if self._waiting[candidate] > 0
                            ]
                        priority_allows = bool(eligible) and index == eligible[0]
                    else:
                        # Preserve the original strict-priority semantics when
                        # no target policy was configured.
                        priority_allows = not any(
                            self._waiting[higher] > 0 for higher in range(index)
                        )
                    if capacity_available and priority_allows:
                        break
                    if not blocked_recorded:
                        self._blocked[index] += 1
                        blocked_recorded = True
                    self._condition.wait(timeout=self._cancel_poll_interval_s)

                self._active_total.value += 1
                self._active[index] += 1
                self._acquired[index] += 1
                self._peak_active_total.value = max(
                    self._peak_active_total.value, self._active_total.value
                )
                self._peak_active[index] = max(
                    self._peak_active[index], self._active[index]
                )
            finally:
                self._waiting[index] -= 1
                self._wait_ns[index] += time.monotonic_ns() - wait_started_ns
                # A cancelled high-priority waiter may have been preventing a
                # lower-priority stage from entering despite spare capacity.
                self._condition.notify_all()

    def release(self, stage: str) -> None:
        """Release one previously acquired slot and wake all priorities."""

        index = self._index(stage)
        with self._condition:
            if self._active[index] <= 0 or self._active_total.value <= 0:
                raise CpuSlotError(
                    f"stage {stage} attempted to release a CPU slot it does not hold"
                )
            self._active[index] -= 1
            self._active_total.value -= 1
            self._released[index] += 1
            self._condition.notify_all()

    def lease(self, stage: str, *, cancel_event: Any = None) -> _CpuSlotLease:
        """Return a context manager that releases its slot even on exceptions."""

        self._index(stage)
        return _CpuSlotLease(self, stage, cancel_event)

    def set_capacity(self, capacity: int) -> None:
        """Change future admission capacity and wake blocked workers.

        Reducing capacity never interrupts active scene tasks.  New acquisitions
        remain blocked until the number of active leases falls below the new cap.
        """

        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("CPU slot capacity must be a positive integer")
        with self._condition:
            if self._targets_enabled.value and sum(self._targets) > capacity:
                raise ValueError(
                    "CPU slot capacity cannot be smaller than configured stage targets "
                    f"({capacity} < {sum(self._targets)})"
                )
            if capacity != self._capacity.value:
                self._capacity_changes.value += 1
            self._capacity.value = capacity
            self._minimum_capacity.value = min(
                self._minimum_capacity.value, capacity
            )
            self._maximum_capacity.value = max(
                self._maximum_capacity.value, capacity
            )
            self._condition.notify_all()

    def set_stage_targets(
        self, stage_targets: Mapping[str, int] | None
    ) -> None:
        """Replace or disable soft per-stage target allocations.

        A stage below its target is preferred only while it has a waiter. If it
        has no work, its share is immediately borrowable according to normal
        stage priority. Changes affect future grants and never preempt active
        scene tasks.
        """

        with self._condition:
            if stage_targets is None:
                enabled = False
                normalized = (0,) * len(self._stages)
            else:
                enabled = True
                normalized = self._normalize_targets(
                    stage_targets, self._capacity.value
                )
            changed = bool(self._targets_enabled.value) != enabled or tuple(
                self._targets
            ) != normalized
            if not changed:
                return
            self._targets_enabled.value = enabled
            for index, target in enumerate(normalized):
                self._targets[index] = target
            self._target_changes.value += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return an internally consistent, JSON-serializable statistics snapshot."""

        with self._condition:
            return {
                "format_version": SNAPSHOT_FORMAT,
                "priority": list(self._stages),
                "capacity": int(self._capacity.value),
                "initial_capacity": int(self._initial_capacity),
                "minimum_capacity": int(self._minimum_capacity.value),
                "maximum_capacity": int(self._maximum_capacity.value),
                "capacity_changes": int(self._capacity_changes.value),
                "targets": (
                    {
                        stage: int(self._targets[index])
                        for index, stage in enumerate(self._stages)
                    }
                    if self._targets_enabled.value
                    else None
                ),
                "target_changes": int(self._target_changes.value),
                "active_total": int(self._active_total.value),
                "peak_active_total": int(self._peak_active_total.value),
                "waiting_total": int(sum(self._waiting)),
                "stages": {
                    stage: {
                        "active": int(self._active[index]),
                        "waiting": int(self._waiting[index]),
                        "peak_active": int(self._peak_active[index]),
                        "acquired": int(self._acquired[index]),
                        "released": int(self._released[index]),
                        "cancelled": int(self._cancelled[index]),
                        "blocked_acquisitions": int(self._blocked[index]),
                        "wait_time_s": float(self._wait_ns[index] / 1_000_000_000),
                    }
                    for index, stage in enumerate(self._stages)
                },
            }


__all__ = [
    "CpuSlotCancelled",
    "CpuSlotError",
    "DEFAULT_STAGE_PRIORITY",
    "PriorityCpuSlots",
    "SNAPSHOT_FORMAT",
    "detect_available_cpu_slots",
]
