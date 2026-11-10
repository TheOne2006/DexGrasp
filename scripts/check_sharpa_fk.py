"""Independent MuJoCo URDF FK check against all stored Sharpa labels."""
import argparse,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py,mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.data.dataset import sha256

def check(root,output):
    hand=URDFHand(root/'hand/hand_spec.json')
    xml=ET.parse(hand.urdf_path).getroot()
    for link in xml.findall('link'):
        for element in list(link):
            if element.tag in ('visual','collision'):link.remove(element)
    mj=xml.find('mujoco')
    if mj is None: mj=ET.SubElement(xml,'mujoco')
    compiler=mj.find('compiler')
    if compiler is None: compiler=ET.SubElement(mj,'compiler')
    compiler.set('fusestatic','false');compiler.set('balanceinertia','true')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));data=mujoco.MjData(model)
    qids=[model.joint(name).qposadr[0] for name in hand.spec['joint_order']]
    ids=[model.body(p['link']).id for p in hand.spec['keypoints']]
    errors=[]
    with h5py.File(root/'shards/part-00000.h5','r') as f:
        for i in range(len(f['grasps/q'])):
            data.qpos[qids]=f['grasps/q'][i];mujoco.mj_kinematics(model,data)
            local=np.array([data.xpos[b]+data.xmat[b].reshape(3,3)@p['local_xyz_m'] for b,p in zip(ids,hand.spec['keypoints'])])
            predicted=local@f['grasps/R'][i].T+f['grasps/t'][i]
            errors.append(float(np.linalg.norm(predicted-f['grasps/K'][i],axis=-1).max()))
    report=dict(engine='MuJoCo '+mujoco.__version__,samples=len(errors),max_error_m=max(errors),mean_max_error_m=float(np.mean(errors)),passed=max(errors)<1e-6,hand_urdf_sha256=sha256(hand.urdf_path),physical_success=None)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();check(a.root,a.output)
