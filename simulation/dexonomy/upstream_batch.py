"""Scene-only batch conversion and six-direction terminal reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.upstream import convert_request
from dexonomy.upstream_pipeline import PipelineError
BATCH_FORMAT = "dexonomyopt-scene-batch/v1"

def _load_manifest(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PipelineError("batch manifest must contain a non-empty cases array")
    entries = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict) or not isinstance(raw.get("request"), str):
            raise PipelineError(f"manifest cases[{index}].request must be a path")
        request_path = Path(raw["request"])
        if not request_path.is_absolute():
            request_path = manifest_path.parent / request_path
        request_path = request_path.resolve()
        if not request_path.is_file():
            raise PipelineError(f"request does not exist: {request_path}")
        entry = dict(raw)
        entry["request"] = str(request_path)
        entries.append(entry)
    return manifest_path, entries



def _load_index(path: str | Path) -> dict[str, Any]:
    index_path = Path(path).resolve()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if data.get("format_version") != BATCH_FORMAT:
        raise PipelineError(f"unsupported batch index: {index_path}")
    return data



def _case_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["scene_id"]: case for case in index["cases"]}



def _scene_id_from_data(data: dict[str, Any], source_path: Path) -> str:
    scene_path = data.get("scene_path")
    if not isinstance(scene_path, str):
        raise PipelineError(f"candidate has no scene_path: {source_path}")
    scene = load_scene_cfg(scene_path)
    scene_id = scene.get("scene_id")
    if not isinstance(scene_id, str):
        raise PipelineError(f"scene has no scene_id: {scene_path}")
    return scene_id


def convert_batch(manifest_path, output_root):
    manifest_path, entries = _load_manifest(manifest_path)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise PipelineError("converted root already exists; use --reuse-converted or a fresh path")
    output_root.mkdir(parents=True)
    cases, seen = [], set()
    for entry in entries:
        request = json.loads(Path(entry['request']).read_text())
        scene_id = request.get('request_id')
        if not isinstance(scene_id, str) or not scene_id or scene_id in seen:
            raise PipelineError('request IDs must be nonempty and unique')
        seen.add(scene_id)
        # Converter validates the ID before writing its scene files.
        from dexonomy.upstream import _SAFE_ID
        if not _SAFE_ID.fullmatch(scene_id):
            raise PipelineError('unsafe scene ID')
        outputs = convert_request(entry['request'], output_root / scene_id)
        report = json.loads(outputs['report'].read_text())
        cases.append({**entry, 'scene_id':scene_id,
                      'grasp_scene':str(outputs['grasp_scene']),
                      'stability_scene':str(outputs['stability_scene']),
                      'runtime_profile':{'target_mass_kg':report['target_mass_kg']}})
    index = {'format_version':BATCH_FORMAT, 'manifest':str(manifest_path),
             'request_count':len(cases), 'cases':cases,
             'grasp_scene_glob':str(output_root / '*' / 'grasp_scene.npy'),
             'batch_index':str(output_root / 'batch_index.json')}
    Path(index['batch_index']).write_text(json.dumps(index,indent=2)+'\n')
    return index


def build_batch_report_from_summaries(batch_index_path, summaries, output_path):
    if set(summaries) != {'init','grasp','stability'}:
        raise PipelineError('Expected only initialization, refinement and six-direction validation')
    cases = _case_map(_load_index(batch_index_path))
    per_scene = {}
    for scene, case in cases.items():
        counts = {key:int(summaries[stage]['per_scene'].get(scene,{}).get('success',0))
                  for key,stage in [('initializations','init'),('refined_grasps','grasp'),('stable_grasps','stability')]}
        if not 0 <= counts['stable_grasps'] <= counts['refined_grasps'] <= counts['initializations']:
            raise PipelineError('Invalid candidate funnel')
        per_scene[scene] = {'object_id':case.get('object_id'), 'label':case.get('label'),
                            'status':'success' if counts['stable_grasps'] else 'failed', **counts}
    totals = {key:sum(r[key] for r in per_scene.values()) for key in ('initializations','refined_grasps','stable_grasps')}
    success = sum(r['stable_grasps']>0 for r in per_scene.values())
    report = {'format_version':'dexonomyopt-stability-report/v1','final_stage':'stability',
              'success_metric':'six_direction_stable_grasp','request_count':len(cases),
              'success_count':success,'success_rate':success/len(cases),
              'gsr':totals['stable_grasps']/totals['initializations'] if totals['initializations'] else 0.,
              'osr':success/len(cases),'totals':totals,'per_scene':per_scene}
    Path(output_path).write_text(json.dumps(report,indent=2)+'\n')
    return report
