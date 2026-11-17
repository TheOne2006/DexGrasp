"""Load native grasps and evaluate a single global pool with frozen defaults.

Object/group labels are descriptive provenance only. All selected records,
including singleton conditions, participate in every valid global pair.
"""
from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import time
import numpy as np

from ._version import __version__
from .data import clear_caches, discover_grasps, load_grasp, register_candidate_identity
from .defaults import configured_parameters, PROFILE_RESOURCE_SHA256
from .features import features_from_records
from .hand import (DEFAULT_KEYPOINT_COUNT, DEFAULT_KEYPOINT_SCHEME,
                   LANDMARK_RESOURCE_SHA256, SharpaHand)
from .metrics import _options, _integer, evaluate_pair_features


def evaluate_dataset(inputs, *, asset_root, config=None, protocol='auto', selection='success',
                     mode='exact', pairs_per_grasp=100, num_pairs=None, seed=0,
                     batch_size=16384, feature_batch_size=1024, temp_dir=None,
                     diagnostics=True, save_features=None, progress=None):
    """Read native records once and evaluate one global feature array.

    Batched FK descriptors are staged as a contiguous temporary file, then read
    into one float64 RAM array. Memory is O(N*D + batch_size*D), never O(N^2).
    No legacy grouped evaluator is called. Input discovery/selection and native
    candidate identity rejection use the same established data contract.
    """
    started = time.monotonic()
    config, profile = configured_parameters(config)
    _options(mode, pairs_per_grasp, num_pairs, seed, batch_size)
    if not isinstance(diagnostics, bool):
        raise ValueError('diagnostics must be boolean')
    feature_batch_size = _integer(feature_batch_size, 'feature_batch_size')
    if selection not in ('success', 'all') or protocol not in ('auto', 'floating', 'tabletop'):
        raise ValueError('Invalid selection or protocol')
    clear_caches()
    feature_path = Path(save_features).expanduser().resolve() if save_features is not None else None
    if feature_path is not None and feature_path.exists():
        raise ValueError('Refusing to overwrite saved global features')
    try:
        paths = discover_grasps(inputs)
    except ValueError as error:
        if str(error) != 'No native grasp .npy files found in the supplied inputs':
            raise
        paths = []
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    hand = None
    dimension = 12 + 3 * DEFAULT_KEYPOINT_COUNT
    records = []
    candidate_sources = {}
    groups = {}
    objects = Counter()
    scales = Counter()
    manifest = hashlib.sha256()
    counts = {'files_discovered': len(paths), 'included_grasps': 0, 'excluded_by_selection': 0}
    excluded_examples = []
    last_update = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='global-grasp-diversity-', dir=temp_dir) as temporary:
        staged = Path(temporary) / 'global_features.f64'
        with staged.open('wb') as stream:
            def flush(batch):
                z = features_from_records(hand, batch, config)
                if z.shape != (len(batch), dimension):
                    raise ValueError('Unexpected native descriptor shape')
                z.astype('<f8', copy=False).tofile(stream)
                for record, feature in zip(batch, z):
                    if record.group_key not in groups:
                        groups[record.group_key] = {'group_key': record.group_key, 'object_key': record.object_key,
                            'scale': list(record.scale), 'hand_name': record.hand_name, 'protocol': record.protocol,
                            'task_key': record.task_key, 'mesh_path': str(record.mesh_path),
                            'object_center_scaled_m': record.object_center.tolist(), 'object_rms_radius_m': record.object_radius,
                            'translation_length_m': config.fixed_ell_t or config.eta_t * record.object_radius, 'n_records': 0}
                    groups[record.group_key]['n_records'] += 1
                    objects[record.object_key] += 1
                    scales[tuple(record.scale)] += 1
                    identity = record.candidate_identity or str(record.path)
                    manifest.update(
                        identity.encode() + b'\0' + str(record.path).encode() + b'\0'
                        + feature.astype('<f8', copy=False).tobytes()
                    )

            for index, path in enumerate(paths):
                record = load_grasp(path, asset_root=asset_root, protocol=protocol, selection=selection)
                if record is None:
                    counts['excluded_by_selection'] += 1
                    if len(excluded_examples) < 20:
                        excluded_examples.append(str(path))
                else:
                    register_candidate_identity(candidate_sources, record.candidate_identity, record.path, scope='selected global dataset')
                    if record.hand_name != 'sharpa_v2':
                        raise ValueError('Only sharpa_v2 is supported')
                    if hand is None:
                        hand = SharpaHand(asset_root)
                        dimension = 12 + 3 * len(hand.keypoint_names)
                    records.append(record)
                    counts['included_grasps'] += 1
                    if len(records) == feature_batch_size:
                        flush(records)
                        records.clear()
                if progress and time.monotonic() - last_update > 10:
                    progress(f'Read {index + 1}/{len(paths)} files; global accepted N={counts["included_grasps"]}')
                    last_update = time.monotonic()
            if records:
                flush(records)
                records.clear()
        z = np.fromfile(staged, dtype='<f8').reshape(counts['included_grasps'], dimension)
    feature_artifact = None
    if feature_path is not None:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with feature_path.open('xb') as stream:
            np.save(stream, z, allow_pickle=False)
        hasher = hashlib.sha256()
        with feature_path.open('rb') as stream:
            while chunk := stream.read(1 << 20):
                hasher.update(chunk)
        feature_artifact = {'path': str(feature_path), 'sha256': hasher.hexdigest(),
                            'shape': list(z.shape), 'weights_already_applied': True}
    if progress:
        progress(f'Global pool N={len(z)}; includes all cross-object and cross-scale nonself pairs')
    feature_seconds = time.monotonic() - started
    metric_started = time.monotonic()
    report = evaluate_pair_features(z, mode=mode, pairs_per_grasp=pairs_per_grasp,
        num_pairs=num_pairs, seed=seed, batch_size=batch_size, diagnostics=diagnostics)
    counts.update(num_groups=len(groups), num_objects=len(objects), grasps_used=len(z),
                  singleton_condition_groups=sum(g['n_records'] == 1 for g in groups.values()))
    u = report['total_U']
    within_groups = sum(
        g['n_records'] * (g['n_records'] - 1) // 2 for g in groups.values()
    )
    within_objects = sum(n * (n - 1) // 2 for n in objects.values())
    within_scales = sum(n * (n - 1) // 2 for n in scales.values())
    report.update(schema='dexonomy-global-pair-diversity/v1', counts=counts,
        feature_config=config.metadata(), hand=hand.metadata() if hand is not None else None,
        feature_manifest_sha256=manifest.hexdigest(),
        saved_global_features=feature_artifact,
        feature_manifest_scope='ordered selected native identity, resolved path and scaled float64 descriptor',
        record_order='sorted resolved discovered paths after selection; no object/group sorting or group-specific seed',
        candidate_identity_policy='reject repeated full native candidate IDs across files; retain equal poses with distinct IDs',
        selection=selection, protocol_request=protocol, excluded_examples=excluded_examples,
        provenance_groups=list(groups.values()),
        pair_population={'total_unordered_pairs': u, 'within_condition_pairs': within_groups,
            'cross_condition_pairs': u - within_groups, 'within_object_pairs': within_objects,
            'cross_object_pairs': u - within_objects, 'same_scale_value_pairs': within_scales,
            'different_scale_value_pairs': u - within_scales,
            'note': 'Overlapping descriptive partitions, not summands of Q and not sampling strata.'},
        coordinate_semantics={'translation': 'Each record is expressed in its own object-local frame, centered at its scaled mesh surface centroid and divided by its own fixed eta*R (or configured absolute length).',
            'rotation': 'Palm orientation relative to its own native object frame; no cross-object axis alignment is estimated.',
            'hand': 'Same Sharpa palm-local landmark coordinates and fixed physical scale for every record.',
            'cross_object_interpretation': 'Global similarity compares these descriptors directly; native mesh-axis conventions affect pose comparisons across objects.'},
        feature_loading_seconds=feature_seconds, metric_seconds=time.monotonic() - metric_started,
        elapsed_seconds=time.monotonic() - started)
    hand_metadata = report['hand'] or {}
    profile['current_keypoint_scheme'] = hand_metadata.get('keypoint_scheme', DEFAULT_KEYPOINT_SCHEME)
    profile['current_keypoint_count'] = hand_metadata.get('keypoint_count', DEFAULT_KEYPOINT_COUNT)
    report['kernel_profile'] = profile
    report['feature_config']['calibration'] = profile['definition']
    report['implementation_version'] = __version__
    report['resource_hashes'] = {'keypoints': LANDMARK_RESOURCE_SHA256,
                                 'frozen_hand50': PROFILE_RESOURCE_SHA256}
    return report
