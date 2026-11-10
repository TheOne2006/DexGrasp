import React from 'react';
const number=(v)=>v==null?'—':v.toFixed(2);
export default function Experiment({runs,summary}) {
  return <section id="experiments" className="experiment"><div className="section-head"><h2>过拟合验证</h2><span>所有结果均来自本机日志与检查点</span></div>
    <div className="experiment-grid"><div>
      {runs.length===0?<p className="muted">1000 条数据实验尚未产生训练指标。</p>:[...runs].sort((a,b)=>({fm:0,flowmap:1,control32_fm:2,control32_refine:3}[a.name]??9)-({fm:0,flowmap:1,control32_fm:2,control32_refine:3}[b.name]??9)).map(run=><Run key={run.name} run={run}/>)}
      <p className="muted small">训练集重建误差用于过拟合诊断；测试集用于观察泛化。关键点误差与 IK 成功均不等于物理抓取成功。</p>
    </div><div><table className="checks"><tbody>
      <tr><th>数据完整性</th><td>{summary.audit?'通过':'准备中'}</td></tr>
      <tr><th>FK 标签一致性</th><td>{summary.audit?'通过':'待检查'}</td></tr>
      <tr><th>独立 MuJoCo FK</th><td>{summary.independent_fk?.passed?'通过':'待检查'}</td></tr>
      <tr><th>软件测试</th><td>{summary.software_tests?.passed?`${summary.software_tests.tests} 项通过`:'待检查'}</td></tr>
      <tr><th>物体级划分</th><td>{summary.audit?'无交叉泄漏':'待检查'}</td></tr>
      <tr><th>掌面射线</th><td>{summary.samples?'每条 ≥64 命中':'待检查'}</td></tr>
      <tr><th>完整手部碰撞</th><td className="muted">未验证</td></tr>
      <tr><th>物理抓取</th><td className="muted">未评估</td></tr>
    </tbody></table><p className="small muted">标签来自 Shadow → Sharpa 显式重定向。原始仿真成功标签不继承到 Sharpa。</p></div></div>
    <details><summary>初始化、冻结策略与 checkpoint</summary><div className="protocol"><p><b>输入场景：</b>来源为浮空抓姿，使用完整 mesh 采样；环境通道加入 z=-0.4 m 的远处人工地面。</p><p><b>FM：</b>随机初始化，物体编码器、场景编码器、Transformer、条件输入与输出头全部训练。</p><p><b>32 条对照：</b>从每个训练物体固定取前两条样本，检查少量样本的过拟合能力；网页另列 800 条训练结果。</p><p><b>FlowMap：</b>复制本轮 FM 权重，新增结束时间输入置零；教师参数及 BN 统计冻结，学生全参数训练；Euler 20 步教师。</p><p><b>存储：</b><code>checkpoints/sharpa_dexonomy_1k_v1/&lt;run&gt;/best.pt、last.pt</code>，包含模型、AdamW、RNG、配置及代码 / 数据 / 手模型哈希。</p><p>过拟合目标：训练集多噪声种子 MPJPE &lt; 5 mm。是否达到，以实测结果为准。</p><a href="/api/protocol" target="_blank" rel="noreferrer">查看完整实验协议 ↗</a></div></details>
  </section>;
}
function Run({run}) {
  const pts=run.metrics.filter(r=>r.train_reconstruction),width=700,height=150;
  const maxX=Math.max(...pts.map(p=>p.step),1),maxY=Math.max(...pts.map(p=>p.train_reconstruction.mpjpe_mm),5)*1.1;
  const path=pts.map((p,i)=>`${i?'L':'M'}${42+p.step/maxX*(width-62)},${height-26-p.train_reconstruction.mpjpe_mm/maxY*(height-42)}`).join(' ');
  return <article className="run"><div className="run-title"><h3>{run.name.includes('control32')?'FM · 32 条固定样本对照'+(run.name.includes('refine')?' / refine':''):run.name==='fm'?'FM · 800 条训练样本':run.name==='flowmap'?'FlowMap · 800 条训练样本':run.name}</h3><span>{run.finished?'已完成':'运行中 / 尚未完成'} · step {run.latest_step}</span></div>
    <svg className="curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${run.name} 训练集关键点误差曲线`}>
      {[0,.5,1].map(t=><g key={t}><line x1="42" y1={height-26-t*(height-42)} x2={width-20} y2={height-26-t*(height-42)} stroke="#e4e9ec"/><text x="3" y={height-22-t*(height-42)}>{(t*maxY).toFixed(0)}</text></g>)}
      <path d={path} fill="none" stroke="#008c7a" strokeWidth="2.5"/><text x="42" y={height-5}>0</text><text x={width-55} y={height-5}>{maxX}</text><text x="300" y={height-5}>optimizer step</text><text x="5" y="12">mm</text>
    </svg><div className="run-values"><span>训练 MPJPE <b>{number(pts.at(-1)?.train_reconstruction.mpjpe_mm)} mm</b></span><span>最佳 <b>{number(run.best_mpjpe)} mm</b></span>{run.evaluation && <span>独立种子评估 <b>{number(run.evaluation.splits.train.mpjpe_mm)} mm</b> · {run.evaluation.overfit_target_met?'达到 5 mm 目标':'尚未达到 5 mm 目标'}</span>}</div>
    {run.evaluation && <table className="results"><thead><tr><th>划分</th><th>样本数</th><th>MPJPE / mm</th><th>P95 / mm</th></tr></thead><tbody>{Object.entries(run.evaluation.splits).map(([key,s])=><tr key={key}><td>{key}</td><td>{s.samples}</td><td>{number(s.mpjpe_mm)}</td><td>{number(s.p95_keypoint_mm)}</td></tr>)}</tbody></table>}
  </article>;
}
