"""Single-consumer shared-memory transport for per-scene candidate bundles.

The streaming pipeline passes only the small descriptor returned by
``publish_scene_bundle`` through ``multiprocessing.Queue``.  The (potentially
large) candidate dictionaries live in one POSIX shared-memory block per scene.

This module deliberately uses ownership transfer instead of distributed
reference counting: a published descriptor has exactly one consumer, and
``consume_scene_bundle`` unlinks its block after copying/deserializing it.
Callers that need to inspect a bundle without taking ownership can use
``read_scene_bundle`` followed by ``unlink_scene_bundle`` in the owner.

Pickle is used because Dexonomy candidate dictionaries contain a mixture of
NumPy arrays, strings, lists, and nested mappings.  Consequently descriptors
must only be accepted from trusted pipeline processes; CRC32 detects accidental
corruption but is not an authenticity check.
"""

from __future__ import annotations

import os
import pickle
import re
import threading
import uuid
import zlib
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Iterable, Mapping


DESCRIPTOR_FORMAT = "dexonomy-scene-shm/v1"
BUNDLE_FORMAT = "dexonomy-scene-bundle/v1"
SHM_NAME_PREFIX = "dexonomy_scene_"
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SHM_NAME_RE = re.compile(
    rf"^{re.escape(SHM_NAME_PREFIX)}([A-Za-z0-9][A-Za-z0-9_-]{{0,63}})_"
    r"([0-9a-f]{32})$"
)
_CRC_RE = re.compile(r"^[0-9a-f]{8}$")
_CREATED_NAMES: dict[str, set[str]] = {}
_CREATED_NAMES_LOCK = threading.Lock()


class SceneShmError(RuntimeError):
    """Base class for scene shared-memory transport failures."""


class SceneShmDescriptorError(SceneShmError, ValueError):
    """The queue descriptor is malformed or inconsistent with its block."""


class SceneShmCorruptionError(SceneShmError):
    """The serialized scene payload failed its integrity or format checks."""


def _validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not _NAMESPACE_RE.fullmatch(namespace):
        raise SceneShmDescriptorError(
            "shared-memory namespace must match "
            "[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
        )
    return namespace


def _expected_name_prefix(namespace: str) -> str:
    return f"{SHM_NAME_PREFIX}{_validate_namespace(namespace)}_"


def _validate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise SceneShmDescriptorError("scene shared-memory descriptor must be a mapping")

    required = {
        "format_version",
        "namespace",
        "scene_id",
        "name",
        "size",
        "crc32",
        "candidate_count",
    }
    missing = required.difference(descriptor)
    if missing:
        raise SceneShmDescriptorError(
            f"scene shared-memory descriptor is missing: {sorted(missing)}"
        )
    if descriptor["format_version"] != DESCRIPTOR_FORMAT:
        raise SceneShmDescriptorError(
            f"unsupported descriptor format: {descriptor['format_version']!r}"
        )

    namespace = _validate_namespace(descriptor["namespace"])
    scene_id = descriptor["scene_id"]
    if not isinstance(scene_id, str) or not scene_id:
        raise SceneShmDescriptorError("descriptor scene_id must be a non-empty string")

    name = descriptor["name"]
    if not isinstance(name, str) or not _SHM_NAME_RE.fullmatch(name):
        raise SceneShmDescriptorError(f"invalid scene shared-memory name: {name!r}")
    if not name.startswith(_expected_name_prefix(namespace)):
        raise SceneShmDescriptorError(
            "descriptor shared-memory name does not belong to its namespace"
        )

    size = descriptor["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SceneShmDescriptorError("descriptor size must be a positive integer")

    crc32 = descriptor["crc32"]
    if not isinstance(crc32, str) or not _CRC_RE.fullmatch(crc32):
        raise SceneShmDescriptorError("descriptor crc32 must be 8 lowercase hex digits")

    candidate_count = descriptor["candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
    ):
        raise SceneShmDescriptorError(
            "descriptor candidate_count must be a non-negative integer"
        )

    return {
        "format_version": DESCRIPTOR_FORMAT,
        "namespace": namespace,
        "scene_id": scene_id,
        "name": name,
        "size": size,
        "crc32": crc32,
        "candidate_count": candidate_count,
    }


def _validate_bundle(bundle: Any, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise SceneShmCorruptionError("scene bundle payload is not a dictionary")
    if bundle.get("format_version") != BUNDLE_FORMAT:
        raise SceneShmCorruptionError(
            f"unsupported scene bundle format: {bundle.get('format_version')!r}"
        )
    if bundle.get("scene_id") != descriptor["scene_id"]:
        raise SceneShmCorruptionError(
            "scene bundle scene_id does not match its descriptor"
        )
    candidates = bundle.get("candidates")
    if not isinstance(candidates, list):
        raise SceneShmCorruptionError("scene bundle candidates must be a list")
    if len(candidates) != descriptor["candidate_count"]:
        raise SceneShmCorruptionError(
            "scene bundle candidate count does not match its descriptor"
        )
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise SceneShmCorruptionError(
            "every scene bundle candidate must be a dictionary"
        )
    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        raise SceneShmCorruptionError("scene bundle metadata must be a dictionary")
    return bundle


def _remember_created(namespace: str, name: str) -> None:
    with _CREATED_NAMES_LOCK:
        _CREATED_NAMES.setdefault(namespace, set()).add(name)


def _forget_created(namespace: str, name: str) -> None:
    with _CREATED_NAMES_LOCK:
        names = _CREATED_NAMES.get(namespace)
        if names is None:
            return
        names.discard(name)
        if not names:
            _CREATED_NAMES.pop(namespace, None)


def publish_scene_bundle(
    scene_id: str,
    candidates: Iterable[dict[str, Any]],
    *,
    namespace: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one complete scene and return its queue-safe descriptor.

    The returned dictionary contains no candidate payload.  It is intended to
    be sent through a multiprocessing queue exactly once.
    """

    namespace = _validate_namespace(namespace)
    if not isinstance(scene_id, str) or not scene_id:
        raise SceneShmDescriptorError("scene_id must be a non-empty string")
    candidate_list = list(candidates)
    if any(not isinstance(candidate, dict) for candidate in candidate_list):
        raise SceneShmDescriptorError("every scene candidate must be a dictionary")
    if metadata is None:
        metadata_dict: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        metadata_dict = dict(metadata)
    else:
        raise SceneShmDescriptorError("scene bundle metadata must be a mapping")

    bundle = {
        "format_version": BUNDLE_FORMAT,
        "scene_id": scene_id,
        "candidates": candidate_list,
        "metadata": metadata_dict,
    }
    payload = pickle.dumps(bundle, protocol=5)
    checksum = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"

    # An explicit, namespaced name lets a crashed batch clean up only its own
    # blocks. UUID collisions are extraordinarily unlikely, but handling one is
    # cheap and keeps creation correct rather than probabilistic.
    shm = None
    name = ""
    for _ in range(10):
        name = f"{_expected_name_prefix(namespace)}{uuid.uuid4().hex}"
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=len(payload))
            break
        except FileExistsError:
            continue
    if shm is None:
        raise SceneShmError("could not allocate a unique scene shared-memory block")

    try:
        shm.buf[: len(payload)] = payload
    except BaseException:
        try:
            shm.unlink()
        finally:
            shm.close()
        raise
    else:
        shm.close()

    _remember_created(namespace, name)
    return {
        "format_version": DESCRIPTOR_FORMAT,
        "namespace": namespace,
        "scene_id": scene_id,
        "name": name,
        "size": len(payload),
        "crc32": checksum,
        "candidate_count": len(candidate_list),
    }


def _load_bundle(
    descriptor: Mapping[str, Any], *, consume: bool
) -> dict[str, Any]:
    validated = _validate_descriptor(descriptor)
    shm = None
    try:
        try:
            shm = shared_memory.SharedMemory(name=validated["name"], create=False)
        except FileNotFoundError as exc:
            raise SceneShmDescriptorError(
                f"scene shared-memory block no longer exists: {validated['name']}"
            ) from exc

        if shm.size != validated["size"]:
            raise SceneShmDescriptorError(
                "descriptor size does not match the shared-memory block "
                f"({validated['size']} != {shm.size})"
            )
        payload = bytes(shm.buf[: validated["size"]])
        actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        if actual_crc != validated["crc32"]:
            raise SceneShmCorruptionError(
                "scene shared-memory CRC mismatch "
                f"({validated['crc32']} != {actual_crc})"
            )
        try:
            bundle = pickle.loads(payload)
        except Exception as exc:
            raise SceneShmCorruptionError(
                "scene shared-memory payload cannot be deserialized"
            ) from exc
        return _validate_bundle(bundle, validated)
    finally:
        if shm is not None:
            if consume:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                else:
                    _forget_created(validated["namespace"], validated["name"])
            shm.close()


def read_scene_bundle(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Copy/deserialise a bundle without unlinking it."""

    return _load_bundle(descriptor, consume=False)


def consume_scene_bundle(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Copy/deserialise a bundle and always unlink its block after attachment.

    Once the descriptor has been validated and its block opened, the block is
    unlinked even when CRC, pickle, or bundle validation fails.  This makes a
    bad payload a reported stage failure rather than a shared-memory leak.
    """

    return _load_bundle(descriptor, consume=True)


def unlink_scene_bundle(
    descriptor: Mapping[str, Any], *, missing_ok: bool = True
) -> bool:
    """Explicitly unlink a published bundle without reading its payload.

    Returns ``True`` when this call removed the block and ``False`` when it was
    already absent.  Only a fully valid, namespaced descriptor can name a block.
    """

    validated = _validate_descriptor(descriptor)
    try:
        shm = shared_memory.SharedMemory(name=validated["name"], create=False)
    except FileNotFoundError:
        _forget_created(validated["namespace"], validated["name"])
        if missing_ok:
            return False
        raise
    try:
        shm.unlink()
    finally:
        shm.close()
    _forget_created(validated["namespace"], validated["name"])
    return True


def _namespace_names(namespace: str) -> set[str]:
    namespace = _validate_namespace(namespace)
    with _CREATED_NAMES_LOCK:
        names = set(_CREATED_NAMES.get(namespace, ()))

    # POSIX shared-memory objects are visible here on Linux.  Keep the local
    # registry as a portable fallback and for environments that hide /dev/shm.
    shm_dir = Path("/dev/shm")
    if os.name == "posix" and shm_dir.is_dir():
        prefix = _expected_name_prefix(namespace)
        try:
            names.update(
                entry.name
                for entry in shm_dir.iterdir()
                if entry.name.startswith(prefix) and _SHM_NAME_RE.fullmatch(entry.name)
            )
        except OSError:
            pass
    return names


def cleanup_scene_shm_namespace(namespace: str) -> list[str]:
    """Best-effort cleanup for all blocks created by one pipeline namespace.

    Namespace validation plus the UUID-suffixed name pattern prevents a batch
    from deleting unrelated shared-memory objects.  The returned list contains
    the names actually unlinked by this call.
    """

    namespace = _validate_namespace(namespace)
    removed = []
    for name in sorted(_namespace_names(namespace)):
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            _forget_created(namespace, name)
            continue
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        else:
            removed.append(name)
        finally:
            shm.close()
            _forget_created(namespace, name)
    return removed


__all__ = [
    "BUNDLE_FORMAT",
    "DESCRIPTOR_FORMAT",
    "SHM_NAME_PREFIX",
    "SceneShmCorruptionError",
    "SceneShmDescriptorError",
    "SceneShmError",
    "cleanup_scene_shm_namespace",
    "consume_scene_bundle",
    "publish_scene_bundle",
    "read_scene_bundle",
    "unlink_scene_bundle",
]
