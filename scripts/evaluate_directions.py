"""Test 26 scene-frame commands per object with fixed scene, gap and initial noise."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dexgrasp.data.dataset import GraspDataset, audit_dataset, sha256, checkpoint_identity_matches
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.geometry.direction import direction_definition, hand_direction
from dexgrasp.geometry.decode import decode_keypoints
from dexgrasp.geometry.mesh import measure_palm_gap, validate_closed_mesh
from dexgrasp.train.runner import load_model, source_state
from dexgrasp.train.objectives import sample_keypoints
from dexgrasp.test.directions import cube_directions, direction_summary


@torch.no_grad()
def evaluate(checkpoint, output, device='cuda:0', seed=50000):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    model,state=load_model(checkpoint,device)
    root=Path(state['config']['dataset_root']);hand=URDFHand(root/'hand/hand_spec.json')
    identity=audit_dataset(root);identity['hand_urdf_sha256']=sha256(hand.urdf_path)
    if not checkpoint_identity_matches(state['data_identity'],identity,root):raise ValueError('Dataset identity mismatch')
    records=json.loads((root/'records.json').read_text())
    chosen={}
    for row in records:chosen.setdefault(row['object_id'],row)
    directions=cube_directions();steps=state['config']['sample_steps']
    protocol=dict(schema='dexgrasp-direction-sweep-v1',evaluation_kind='direction_sweep',
                  label=('手腕 +Zh · 26 方向测试' if direction_definition(hand.spec)['kind']=='official_wrist_positive_z_v2' else '历史掌面法向 · 26 方向测试'),run=output.name,dataset=root.name,dataset_root=str(root),
                  created_at=datetime.now(timezone.utc).isoformat(),source=source_state(),data_identity=identity,
                  checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint),optimizer_steps=state['step'],
                  method=model.method,seed=seed,seeds=1,sampling_steps=steps,directions=directions,direction_count=26,
                  model_variant=model.cfg.variant,palm_gap_conditioned=model.cfg.use_gap,
                  direction_frame=direction_definition(hand.spec)['kind'],
                  direction_definition=direction_definition(hand.spec),
                  selection='first record per object in records.json order; no ranking',
                  samples_per_object=1,candidates_per_direction=1,expected_cases=len(chosen)*26,
                  selected_samples=[dict(id=r['id'],object_id=r['object_id'],split=r['split']) for r in chosen.values()],
                  controls='same scene point clouds, gap and initial noise across all 26 commands and original-direction baseline for each object',
                  noise='CPU torch.Generator(seed + object index); normal [1,21,3], repeated without resampling',
                  ik_tolerance_m=.003,ik_max_nfev=100,direction_tolerance_deg=10.,gap_tolerance_m=.005,
                  ik_initialization='joint limit midpoint, no reference q',ranking='none; failures retained',
                  physical_success_rate=None,
                  note='Direction-response diagnostic, one pose/scene per object. Fixed gap may be infeasible for some commands. No paired target MPJPE for changed commands; no collision or physics verdict.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    datasets={split:GraspDataset(root,split,model.cfg.coord_scale_m) for split in ('train','test')}
    indices={split:{row['grasp_index']:i for i,row in enumerate(ds.rows)} for split,ds in datasets.items()}
    results=[];baselines=[];arrays={};started=time.monotonic()
    try:
        with (output/'progress.jsonl').open('x') as log:
            for oi,record in enumerate(chosen.values()):
                batch=datasets[record['split']][indices[record['split']][record['grasp_index']]]
                commands=directions+[dict(direction_id='reference',direction_label='原标注方向',direction_group='reference',
                                          command_direction=batch['condition']['grasp_direction_cmd'].tolist())]
                n=len(commands)
                condition={k:batch['condition'][k].unsqueeze(0).repeat(n,*([1]*batch['condition'][k].ndim)).to(device)
                           for k in model.cfg.condition_keys}
                condition['grasp_direction_cmd']=torch.tensor([c['command_direction'] for c in commands],dtype=torch.float32,device=device)
                noise=torch.randn((1,21,3),generator=torch.Generator().manual_seed(seed+oi))
                generated=sample_keypoints(model,condition,steps=steps,noise=noise.repeat(n,1,1).to(device))
                world=(generated*model.cfg.coord_scale_m+batch['metadata']['center_m'].to(device)[None,None]).cpu().numpy()
                arrays[record['id']+'_noise']=noise.numpy();arrays[record['id']+'_keypoints']=world
                with np.load(root/'assets'/f"{record['id']}.npz",allow_pickle=False) as mesh:
                    vertices,faces=mesh['vertices'],mesh['faces']
                validate_closed_mesh(vertices,faces)
                for command,k in zip(commands,world):
                    case_id=hashlib.sha256(f"{record['id']}:{command['direction_id']}:{seed+oi}".encode()).hexdigest()[:20]
                    row=dict(id=case_id,sample_id=record['id'],object_id=record['object_id'],split=record['split'],**command,
                             seed=seed+oi,requested_palm_gap_m=float(batch['condition']['palm_gap'][0]),
                             noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),keypoints=k.tolist(),
                             physical_success=None,collision_free=None,ik_success=False,geometric_pass=False)
                    try:
                        decoded=decode_keypoints(k,hand,np.array(command['command_direction']),tolerance_m=.003,max_nfev=100)
                        row.update({key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in decoded.items()})
                        row.update(ik_rmse_mm=decoded['reconstruction_error_m']*1000,palm_rmse_mm=decoded['palm_registration_error_m']*1000,
                                   output_direction=hand_direction(decoded['R'],hand.spec).tolist())
                        gap=measure_palm_gap(vertices,faces,decoded['R'],decoded['t'],hand.spec['palm'],check_mesh=False)
                        gap_error=abs(gap['h']-row['requested_palm_gap_m']) if gap['valid'] else None
                        row.update(palm_gap_valid=gap['valid'],palm_gap_invalid_reason=gap['reason'],palm_hit_count=gap['hit_count'],
                                   measured_palm_gap_m=gap['h'],palm_gap_error_mm=gap_error*1000 if gap_error is not None else None,
                                   geometric_pass=bool(decoded['ik_success'] and decoded['joint_limits_valid'] and gap['valid']
                                                      and gap_error<=.005 and decoded['direction_error_deg']<=10.))
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:row['error']=str(exc)
                    (baselines if command['direction_id']=='reference' else results).append(row)
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                print(f"{oi+1}/{len(chosen)} objects: {record['object_id']} ({record['split']}); {len(results)} directions decoded",flush=True)
        report=dict(**protocol,seconds=time.monotonic()-started,records=results,baseline_records=baselines,
                    summaries=[direction_summary(results,s) for s in ('all','train','test')],
                    baseline_summaries=[direction_summary(baselines,s) for s in ('all','train','test')])
        np.savez_compressed(output/'samples.npz',**arrays)
        temporary=output/'evaluation.partial';temporary.write_text(json.dumps(report,allow_nan=False)+'\n');temporary.replace(output/'evaluation.json')
        print(json.dumps(dict(summaries=report['summaries'],baseline_summaries=report['baseline_summaries']),indent=2),flush=True)
    finally:
        for dataset in datasets.values():dataset.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=50000)
    a=p.parse_args();torch.set_num_threads(4);evaluate(a.checkpoint,a.output,a.device,a.seed)
