import React, {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export default function Viewer({sample, shadow, predicted}) {
  const node = useRef(null), [error,setError] = useState('');
  useEffect(() => {
    if (!sample || !node.current) return;
    const host=node.current;
    let renderer;
    try { renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); }
    catch { setError('浏览器无法启用 WebGL，请启用硬件加速后重试。'); return; }
    const scene=new THREE.Scene(); scene.background=new THREE.Color('#f3f6f8');
    const camera=new THREE.PerspectiveCamera(38,1,.001,10); camera.up.set(0,0,1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));host.appendChild(renderer.domElement);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
    scene.add(new THREE.HemisphereLight(0xffffff,0x717981,2.4));
    const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(.3,-.4,.7);scene.add(light);
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(sample.vertices.flat(),3));
    geometry.setIndex(sample.faces.flat());geometry.computeVertexNormals();
    const material=new THREE.MeshStandardMaterial({color:0xb2bec5,roughness:.8,metalness:.08,side:THREE.DoubleSide,transparent:true,opacity:.55,depthWrite:false});
    scene.add(new THREE.Mesh(geometry,material));
    function skeleton(points,color,radius) {
      const mat=new THREE.MeshStandardMaterial({color,roughness:.35});
      const ball=new THREE.SphereGeometry(radius,12,8);
      points.forEach(p=>{const s=new THREE.Mesh(ball,mat);s.position.fromArray(p);scene.add(s);});
      const pairs=[];
      for(let f=0;f<5;f++){const base=1+4*f;pairs.push([0,base],[base,base+1],[base+1,base+2],[base+2,base+3]);}
      pairs.forEach(([a,b])=>{const start=new THREE.Vector3().fromArray(points[a]),end=new THREE.Vector3().fromArray(points[b]);
        const g=new THREE.CylinderGeometry(radius*.52,radius*.52,start.distanceTo(end),8);
        const line=new THREE.Mesh(g,mat);line.position.copy(start).add(end).multiplyScalar(.5);line.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),end.sub(start).normalize());scene.add(line);});
    }
    skeleton(sample.K,0x008c7a,.0028);
    if(shadow)skeleton(sample.shadow_K,0xea8b2e,.002);
    for(const [method,color] of [['fm',0x2877d4],['flowmap',0x655bc8]]) {
      if(predicted[method] && sample.predictions?.[method])
        skeleton(sample.predictions[method].keypoints,color,.0018);
    }
    const box=new THREE.Box3().setFromObject(scene),center=box.getCenter(new THREE.Vector3()),span=box.getSize(new THREE.Vector3()).length();
    const grid=new THREE.GridHelper(.5,10,0xc5cfd6,0xe2e8ec);grid.rotation.x=Math.PI/2;grid.position.set(center.x,center.y,box.min.z-.015);scene.add(grid);
    const axes=new THREE.AxesHelper(.05);axes.position.copy(grid.position);scene.add(axes);
    controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.75,-1,.65).multiplyScalar(Math.max(span,.18)*1.1));controls.update();
    const render=()=>renderer.render(scene,camera);
    const resize=()=>{camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight);render();};
    const observer=new ResizeObserver(resize);observer.observe(host);controls.addEventListener('change',render);resize();
    return ()=>{observer.disconnect();controls.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material) o.material.dispose();});renderer.dispose();renderer.domElement.remove();};
  },[sample,shadow,predicted]);
  return <div className="viewer" ref={node} aria-label="三维抓姿视图">{error && <p role="alert">{error}</p>}{!sample && <div className="empty">选择样本以查看三维抓姿</div>}<div className="view-help">拖动旋转 · 滚轮缩放 · 右键平移　/　单位 m</div></div>;
}
