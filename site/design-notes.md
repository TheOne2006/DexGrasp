# Explorer design

Reference: `outputs/sharpa_dexonomy_1k/design/concept.png` (built-in Image Gen, complete primary-screen UI concept; a design reference, never scientific evidence).

White background, charcoal text, teal #008c7a Sharpa, orange #ea8b2e Shadow, pale #f3f6f8 plotting canvas. Header 64px; 28px content gutters; 34px title / 17px subtitle; thin gray dividers; 4–6px control radii. Object rail → interactive scientific canvas → sample inspector. Experiment strip follows. React + Vite; Three.js on-demand import; real data read from read-only Python API. Model and source objects rendered from numerical geometry, never generated imagery.

Allowed copy and regions: DexGrasp; 数据集; 实验验证; Sharpa; 数据集与重定向; 现有物体与抓姿 · Sharpa 运动学诊断; sample/object/split counts; physical success unassessed; search/filter; sample ID/index; retarget error; palm rays; joint bounds; overfit curves; integrity/FK/physical checks.

Intentional changes from concept: numerical results replace em-dash placeholders once measured; added per-sample controls and provenance links; no meaningless hand-picker or decorative toolbar; anatomical landmark errors replace unavailable orientation/joint cross-hand errors; native 3D axes and real object replace concept illustration. Mobile stacks the object rail, viewport and inspector; scientific content remains the same.
