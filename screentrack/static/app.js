'use strict';

const STATE = {
  clips: [], jobs: [], clip: null, job: null, tab: 'anchor',
  anchorFrame: 0, quad: null, dragging: -1, frameIdx: 0,
  track: null, truth: null, evalData: null, pollTimer: null, options: {
    draw_features: true, draw_rejected: true, smooth: true,
  },
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, children = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    e.append(c.nodeType ? c : document.createTextNode(c));
  }
  return e;
};
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
};
const toast = (msg, ms = 2600) => {
  const t = $('#toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(t._h); t._h = setTimeout(() => (t.style.display = 'none'), ms);
};
const fmt = (v, d = 2) => (v == null || Number.isNaN(v)) ? 'n/a' : Number(v).toFixed(d);

const STATE_COLORS = {
  anchor: '#37d67a', tracking: '#37d67a', tracking_partial: '#ffb020',
  weak: '#ffb020', predicted: '#4ea1ff', paused: '#ff5c5c', reacquired: '#e05aff',
};

async function loadClips() {
  const data = await api('/api/clips');
  STATE.clips = data.clips;
  renderClips();
  if (!STATE.clip && data.clips.length) selectClip(data.clips[0].name);
}
async function loadJobs() {
  const data = await api('/api/jobs');
  STATE.jobs = data.jobs;
  renderJobs();
}
function renderClips() {
  const list = $('#clip-list');
  list.innerHTML = '';
  for (const c of STATE.clips) {
    const item = el('div', {
      class: 'clip-item' + (STATE.clip === c.name ? ' active' : ''),
      onclick: () => selectClip(c.name),
    }, [
      el('div', { class: 'name' }, [c.name, el('span', { class: 'badge' }, [`${c.frames}f`])]),
      el('div', { class: 'desc' }, [c.description || '']),
    ]);
    list.append(item);
  }
}
function renderJobs() {
  const list = $('#job-list');
  list.innerHTML = '';
  if (!STATE.jobs.length) {
    list.append(el('div', { class: 'hint' }, ['No jobs yet. Pick a clip and place an anchor.']));
  }
  for (const j of STATE.jobs) {
    const badgeClass = j.status === 'ready' ? 'ok' : j.status === 'running' ? 'run' : j.status === 'failed' ? 'err' : '';
    const item = el('div', {
      class: 'job-item' + (STATE.job && STATE.job.id === j.id ? ' active' : ''),
      onclick: () => openJob(j.id),
    }, [
      el('div', { class: 'name' }, [j.name, el('span', { class: 'badge ' + badgeClass }, [j.status])]),
      el('div', { class: 'desc' }, [j.has_track ? 'track ✓' : 'no track',
        j.has_eval ? ' · eval ✓' : '', j.clip ? ` · ${j.clip}` : '']),
    ]);
    list.append(item);
  }
}

async function selectClip(name) {
  STATE.clip = name; STATE.tab = 'anchor'; STATE.job = null;
  STATE.track = null; STATE.truth = null; STATE.evalData = null;
  STATE.quad = null; STATE.anchorFrame = 0; STATE.frameIdx = 0;
  renderClips();
  const truth = await api(`/api/clips/${name}/truth`);
  STATE.truth = truth;
  STATE.quad = truth.frames[0].quad.map((p) => [...p]);
  renderWorkspace();
}

async function openJob(id) {
  STATE.job = STATE.jobs.find((j) => j.id === id) || await api(`/api/jobs/${id}`);
  STATE.tab = STATE.job.status === 'ready' ? 'review' : 'anchor';
  if (STATE.job.anchor) {
    STATE.quad = STATE.job.anchor.quad.map((p) => [...p]);
    STATE.anchorFrame = STATE.job.anchor.frame;
    if (STATE.job.clip) STATE.clip = STATE.job.clip;
  }
  await refreshJobData();
  renderJobs(); renderWorkspace();
  if (STATE.job.status === 'running' || STATE.job.status === 'created') startPolling(id);
}

async function refreshJobData() {
  const j = STATE.job;
  if (!j) return;
  try {
    STATE.track = await api(`/api/jobs/${j.id}/track`);
    const t = await api(`/api/jobs/${j.id}/truth`);
    if (t.truth) STATE.truth = t.truth;
    try { STATE.evalData = await api(`/api/jobs/${j.id}/evaluation`); } catch (e) { STATE.evalData = null; }
  } catch (e) { /* track not ready yet */ }
}

function startPolling(id) {
  clearInterval(STATE.pollTimer);
  STATE.pollTimer = setInterval(async () => {
    try {
      const list = await api('/api/jobs');
      STATE.jobs = list.jobs;
      const j = list.jobs.find((x) => x.id === id);
      if (!j) return;
      STATE.job = j;
      renderJobs();
      updateProgress();
      if (j.status === 'ready' || j.status === 'failed') {
        clearInterval(STATE.pollTimer);
        await refreshJobData();
        if (j.status === 'ready') { STATE.tab = 'review'; }
        renderWorkspace();
      }
    } catch (e) { clearInterval(STATE.pollTimer); }
  }, 1200);
}

function updateProgress() {
  const p = document.querySelector('#job-progress');
  if (!p || !STATE.job) return;
  const bar = p.querySelector('.progress-line > div');
  if (bar) bar.style.width = `${Math.round((STATE.job.progress || 0) * 100)}%`;
  const label = p.querySelector('.stage');
  if (label) label.textContent = `${STATE.job.stage} (${Math.round((STATE.job.progress || 0) * 100)}%)`;
}

/* ---------------- anchor view ---------------- */

function renderWorkspace() {
  const ws = $('#workspace');
  ws.innerHTML = '';
  if (!STATE.clip) { ws.append(el('div', { class: 'panel' }, ['Select a clip to begin.'])); return; }
  const tabs = el('div', { class: 'tabs' });
  for (const [key, label] of [['anchor', 'Anchor & track'], ['review', 'Review & evidence'],
                              ['output', 'Output video'], ['report', 'Report']]) {
    tabs.append(el('div', {
      class: 'tab' + (STATE.tab === key ? ' active' : ''),
      onclick: () => { STATE.tab = key; renderWorkspace(); },
    }, [label]));
  }
  ws.append(tabs);
  if (STATE.tab === 'anchor') renderAnchor(ws);
  else if (STATE.tab === 'review') renderReview(ws);
  else if (STATE.tab === 'output') renderOutput(ws);
  else renderReport(ws);
}

async function renderAnchor(ws) {
  const clip = STATE.clips.find((c) => c.name === STATE.clip);
  const job = STATE.job;
  const right = el('div', { class: 'panel' });
  const grid = el('div', { class: 'viewer' });
  const wrap = el('div', { class: 'canvas-wrap' });
  const img = el('img', { id: 'anchor-img', alt: 'anchor frame' });
  const canvas = el('canvas', { class: 'overlay', id: 'anchor-canvas' });
  wrap.append(img, canvas);
  grid.append(wrap, right);
  ws.append(grid);

  right.append(el('h3', {}, ['1 · Choose the anchor frame']));
  const fi = el('input', {
    type: 'range', min: 0, max: Math.max(0, (clip.frames || 1) - 1), value: STATE.anchorFrame,
    class: 'slider', id: 'anchor-frame',
    oninput: (e) => { STATE.anchorFrame = +e.target.value; loadAnchorFrame(); },
  });
  right.append(el('div', { class: 'row' }, [fi, el('span', { id: 'anchor-frame-label', class: 'num' }, [String(STATE.anchorFrame)])]));
  right.append(el('div', { class: 'hint' }, ['Drag the four corners on the image, or press “use ground-truth quad” to load a known-good anchor.']));

  right.append(el('h3', {}, ['2 · Screen corners (frame coordinates)']));
  const kv = el('div', { class: 'kv', id: 'quad-kv' });
  right.append(kv);

  right.append(el('div', { class: 'row', style: 'margin-top:10px' }, [
    el('button', { class: 'small', onclick: () => { STATE.quad = STATE.truth.frames[STATE.anchorFrame].quad.map((p) => [...p]); drawAnchor(); } }, ['Use truth quad']),
    el('button', { class: 'small', onclick: () => { STATE.quad = inflate(STATE.truth.frames[STATE.anchorFrame].quad); drawAnchor(); } }, ['Reset to ±6% estimate']),
  ]));

  right.append(el('h3', { style: 'margin-top:12px' }, ['3 · Run']));
  const nameInput = el('input', { type: 'text', value: job ? job.name : `${STATE.clip} · run ${STATE.jobs.length + 1}`, style: 'width:100%' });
  right.append(el('div', { class: 'row' }, [nameInput]));
  const runBtn = el('button', { class: 'primary', onclick: () => startJob(nameInput.value) }, ['Track, composite & evaluate']);
  right.append(el('div', { class: 'row' }, [runBtn]));
  const prog = el('div', { id: 'job-progress' }, []);
  right.append(prog);

  if (job && job.status === 'failed' && job.error) {
    right.append(el('div', { class: 'alert' }, [job.error]));
  }
  if (job && (job.status === 'running' || job.status === 'created')) {
    prog.append(el('div', { class: 'stage' }, [job.stage]),
      el('div', { class: 'progress-line' }, [el('div')]));
    setTimeout(updateProgress, 0);
  }
  if (job && job.status === 'ready') {
    right.append(el('div', { class: 'ok-box' }, ['Ready. Open “Review & evidence”.']));
  }

  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; drawAnchor(); };
  loadAnchorFrame();
  bindCanvas(canvas);
}

function inflate(quad, frac = 0.06) {
  const c = quad.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]);
  return quad.map((p) => [c[0] + (p[0] - c[0]) * (1 + frac), c[1] + (p[1] - c[1]) * (1 + frac)]);
}

async function loadAnchorFrame() {
  const idx = STATE.anchorFrame;
  const data = await api(`/api/clips/${STATE.clip}/frame/${idx}`);
  const img = $('#anchor-img');
  if (!img) return;
  img.src = 'data:image/jpeg;base64,' + data.jpeg_b64;
  const lbl = $('#anchor-frame-label'); if (lbl) lbl.textContent = String(idx);
  const slider = $('#anchor-frame'); if (slider) slider.value = String(idx);
}

function bindCanvas(canvas) {
  const toImg = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [ (ev.clientX - r.left) * canvas.width / r.width,
             (ev.clientY - r.top) * canvas.height / r.height ];
  };
  canvas.addEventListener('pointerdown', (ev) => {
    if (!STATE.quad) return;
    const [x, y] = toImg(ev);
    let best = -1, bd = 22 * canvas.width / canvas.getBoundingClientRect().width;
    STATE.quad.forEach((p, i) => {
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d < bd) { bd = d; best = i; }
    });
    STATE.dragging = best;
    if (best >= 0) canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (STATE.dragging < 0 || !STATE.quad) return;
    const [x, y] = toImg(ev);
    STATE.quad[STATE.dragging] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    drawAnchor();
  });
  canvas.addEventListener('pointerup', () => { STATE.dragging = -1; });
  canvas.addEventListener('pointercancel', () => { STATE.dragging = -1; });
}

function drawAnchor() {
  const canvas = $('#anchor-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (STATE.truth) {
    const gt = STATE.truth.frames[STATE.anchorFrame].quad;
    ctx.strokeStyle = 'rgba(255,255,0,.7)'; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); gt.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  }
  if (!STATE.quad) return;
  ctx.strokeStyle = '#4ea1ff'; ctx.lineWidth = 2;
  ctx.beginPath(); STATE.quad.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.closePath(); ctx.stroke();
  STATE.quad.forEach((p, i) => {
    ctx.fillStyle = '#4ea1ff';
    ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '11px monospace';
    ctx.fillText(['TL','TR','BR','BL'][i], p[0] + 8, p[1] - 6);
  });
  const kv = $('#quad-kv');
  if (kv) {
    kv.innerHTML = '';
    STATE.quad.forEach((p, i) => {
      kv.append(el('div', { class: 'k' }, [['TL','TR','BR','BL'][i]]),
                el('div', { class: 'num' }, [`${p[0].toFixed(1)}, ${p[1].toFixed(1)}`]));
    });
  }
}

async function startJob(name) {
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ name, clip: STATE.clip }),
    });
    STATE.job = job;
    STATE.jobs.unshift(job);
    await api(`/api/jobs/${job.id}/track`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_frame: STATE.anchorFrame, quad: STATE.quad, options: STATE.options,
      }),
    });
    renderJobs(); renderWorkspace();
    startPolling(job.id);
    toast('Tracking started');
  } catch (e) { toast('Failed: ' + e.message, 5000); }
}

/* ---------------- review view ---------------- */

function recForFrame(idx) {
  if (!STATE.track) return null;
  return STATE.track.frames.find((f) => f.frame === idx) || null;
}

async function renderReview(ws) {
  if (!STATE.track) {
    ws.append(el('div', { class: 'panel' }, ['No track yet for this job.']));
    return;
  }
  const total = STATE.track.frames.length;
  if (!STATE.frameIdx || STATE.frameIdx >= total) STATE.frameIdx = 0;
  const f0 = STATE.track.frames[STATE.frameIdx].frame;
  const grid = el('div', { class: 'viewer' });
  const wrap = el('div', { class: 'canvas-wrap' });
  const img = el('img', { id: 'review-img' });
  const canvas = el('canvas', { class: 'overlay', id: 'review-canvas', style: 'cursor:default' });
  wrap.append(img, canvas);
  grid.append(wrap, el('div', { class: 'panel', id: 'review-side' }));
  ws.append(grid);

  const controls = el('div', { class: 'panel', style: 'margin-top:12px' });
  controls.append(el('div', { class: 'row' }, [
    el('button', { class: 'small', onclick: () => step(-1) }, ['◀ prev']),
    el('button', { class: 'small', onclick: () => step(1) }, ['next ▶']),
    el('span', { class: 'muted' }, ['frame']),
    el('span', { id: 'frame-label', class: 'num' }, [String(f0)]),
    el('span', { class: 'muted' }, ['/ ' + (total - 1)]),
    el('label', {}, [el('input', { type: 'checkbox', checked: true, onchange: (e) => { STATE.options.draw_features = e.target.checked; drawReview(); } }), ' features']),
    el('label', {}, [el('input', { type: 'checkbox', checked: true, onchange: (e) => { STATE.options.draw_rejected = e.target.checked; drawReview(); } }), ' rejected']),
    el('label', {}, [el('input', { type: 'checkbox', checked: true, onchange: (e) => { STATE.options.show_gt = e.target.checked; drawReview(); } }), ' truth quad']),
  ]));
  const slider = el('input', {
    type: 'range', min: 0, max: total - 1, value: STATE.frameIdx, class: 'slider',
    oninput: (e) => { STATE.frameIdx = +e.target.value; syncReview(); },
  });
  controls.append(slider);
  controls.append(el('div', { id: 'timeline', class: 'timeline' }));
  controls.append(el('div', { class: 'legend' }, [
    leg('#37d67a', 'tracking'), leg('#ffb020', 'partial / weak'), leg('#4ea1ff', 'predicted'),
    leg('#ff5c5c', 'paused'), leg('#e05aff', 'reacquired'), leg('#2ee6d6', 'inlier'),
    leg('#ff5c5c', 'rejected match'),
  ]));
  ws.append(controls);
  drawTimeline();
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; drawReview(); };
  await syncReview();
  slider.addEventListener('input', () => {
    document.querySelectorAll('.timeline .cell').forEach((c, i) => {
      c.style.outline = i === STATE.frameIdx ? '1px solid #fff' : 'none';
    });
  });
}
function leg(color, text) {
  return el('span', {}, [el('i', { style: `background:${color}` }), text]);
}
function step(d) {
  STATE.frameIdx = Math.max(0, Math.min(STATE.track.frames.length - 1, STATE.frameIdx + d));
  syncReview();
}
async function syncReview() {
  const rec = STATE.track.frames[STATE.frameIdx];
  if (!rec) return;
  const data = await api(`/api/jobs/${STATE.job.id}/frame/${rec.frame}`);
  const img = $('#review-img');
  if (img) img.src = 'data:image/jpeg;base64,' + data.jpeg_b64;
  const lbl = $('#frame-label'); if (lbl) lbl.textContent = String(rec.frame);
  const slider = document.querySelector('.panel input[type=range].slider');
  if (slider && !document.activeElement?.matches('input[type=range]:not(.slider)')) slider.value = String(STATE.frameIdx);
  drawReview();
  drawSide(rec);
  document.querySelectorAll('.timeline .cell').forEach((c, i) => {
    c.style.outline = i === STATE.frameIdx ? '1px solid #fff' : 'none';
  });
}
function drawReview() {
  const canvas = $('#review-canvas');
  if (!canvas || !STATE.track) return;
  const rec = STATE.track.frames[STATE.frameIdx];
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const showGt = STATE.options.show_gt !== false;
  if (showGt && STATE.truth) {
    const gt = STATE.truth.frames[rec.frame];
    if (gt) {
      ctx.strokeStyle = 'rgba(255,255,0,.8)'; ctx.setLineDash([6, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); gt.quad.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  if (rec.predicted_quad && ['predicted','paused','weak','tracking_partial'].includes(rec.state)) {
    drawQuad(ctx, rec.predicted_quad, 'rgba(78,161,255,.55)', 1, [4, 3]);
  }
  if (STATE.options.draw_features !== false) {
    ctx.fillStyle = '#2ee6d6';
    for (const [x, y] of rec.inliers) ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
  if (STATE.options.draw_rejected !== false) {
    ctx.strokeStyle = '#ff5c5c'; ctx.lineWidth = 1.2;
    for (const r of rec.rejected) {
      const [x, y] = r.p; ctx.beginPath();
      ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4);
      ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke();
    }
  }
  if (rec.quad) drawQuad(ctx, rec.quad, STATE_COLORS[rec.state] || '#fff', 2, []);
}
function drawQuad(ctx, quad, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath(); quad.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
}
function drawSide(rec) {
  const side = $('#review-side');
  if (!side) return;
  side.innerHTML = '';
  const truthFrame = STATE.truth ? STATE.truth.frames[rec.frame] : null;
  let err = null;
  if (rec.quad && truthFrame) {
    err = rec.quad.reduce((a, p, i) => a + Math.hypot(p[0] - truthFrame.quad[i][0], p[1] - truthFrame.quad[i][1]), 0) / 4;
  }
  side.append(el('h3', {}, ['Observation']));
  const kv = el('div', { class: 'kv' });
  const rows = [
    ['state', rec.state], ['decision', rec.decision || '—'],
    ['inliers', rec.inlier_count], ['rejected', rec.rejected.length],
    ['coverage', `${Math.round((rec.coverage || 0) * 100)}%`],
    ['reproj (median)', `${fmt(rec.reproj_median)} px`],
    ['occlusion (est)', rec.occlusion == null ? 'n/a' : `${Math.round(rec.occlusion * 100)}%`],
    ['confidence', `${fmt(rec.confidence, 0)} / 100`],
    ['corner σ', (rec.corner_sigma || []).map((s) => fmt(s, 1)).join(', ')],
    ['corners verified', (rec.corner_visible || []).filter(Boolean).length + ' / 4'],
    ['keyframe', rec.keyframe ?? '—'],
  ];
  if (truthFrame) {
    rows.push(['GT occlusion', `${Math.round(truthFrame.occlusion * 100)}%`]);
    rows.push(['GT corners visible', truthFrame.corner_visible.filter(Boolean).length + ' / 4']);
    rows.push(['corner error', err == null ? 'overlay withheld' : `${fmt(err)} px`]);
  }
  for (const [k, v] of rows) kv.append(el('div', { class: 'k' }, [k]), el('div', { class: 'num' }, [String(v)]));
  side.append(kv);
  if (rec.note) side.append(el('div', { class: 'hint' }, [rec.note]));
  if (rec.reject_counts && Object.keys(rec.reject_counts).length) {
    side.append(el('h3', { style: 'margin-top:10px' }, ['Rejected matches']));
    const rk = el('div', { class: 'kv' });
    for (const [k, v] of Object.entries(rec.reject_counts)) rk.append(el('div', { class: 'k' }, [k]), el('div', { class: 'num' }, [String(v)]));
    side.append(rk);
  }
}
function drawTimeline() {
  const tl = $('#timeline');
  if (!tl || !STATE.track) return;
  tl.innerHTML = '';
  tl.style.gridTemplateColumns = `repeat(${Math.min(60, STATE.track.frames.length)}, 1fr)`;
  for (const rec of STATE.track.frames) {
    const cell = el('div', {
      class: 'cell', title: `frame ${rec.frame}: ${rec.state}`,
      style: `background:${STATE_COLORS[rec.state] || '#555'};height:11px`,
    });
    cell.addEventListener('click', () => {
      STATE.frameIdx = STATE.track.frames.findIndex((f) => f.frame === rec.frame);
      syncReview();
    });
    tl.append(cell);
  }
}

/* ---------------- output view ---------------- */

function renderOutput(ws) {
  if (!STATE.job || STATE.job.status !== 'ready') {
    ws.append(el('div', { class: 'panel' }, ['Run a tracking job first.']));
    return;
  }
  const row = el('div', { class: 'grid2' });
  row.append(videoPanel('Output composite', `/api/jobs/${STATE.job.id}/video/composite`));
  row.append(videoPanel('Diagnostics (features, rejected matches, state)', `/api/jobs/${STATE.job.id}/video/diagnostics`));
  ws.append(row);
  ws.append(el('div', { class: 'panel', style: 'margin-top:12px' }, [
    el('h3', {}, ['Source video (unchanged)']),
    el('video', { src: `/api/jobs/${STATE.job.id}/video/source`, controls: 'controls', style: 'width:100%' }),
    el('div', { class: 'hint' }, ['Tracking and composites are separate files; the source is never overwritten, so failed frames can be replayed as-is.']),
  ]));
}
function videoPanel(title, src) {
  return el('div', { class: 'panel' }, [
    el('h3', {}, [title]),
    el('video', { src, controls: 'controls', style: 'width:100%' }),
  ]);
}

/* ---------------- report view ---------------- */

function renderReport(ws) {
  if (!STATE.job || STATE.job.status !== 'ready') {
    ws.append(el('div', { class: 'panel' }, ['Run a tracking job to see the report.']));
    return;
  }
  const ev = STATE.evalData;
  const track = STATE.track;
  const panel = el('div', { class: 'panel' });
  panel.append(el('h3', {}, ['Tracking summary']));
  const stats = track.stats || {};
  const s = el('div', { class: 'kv' });
  for (const [k, v] of Object.entries(stats)) s.append(el('div', { class: 'k' }, [k]), el('div', { class: 'num' }, [String(v)]));
  panel.append(s);
  if (ev) {
    panel.append(el('h3', { style: 'margin-top:12px' }, ['Ground-truth comparison']));
    const m = ev.metrics;
    const mk = el('div', { class: 'kv' });
    const add = (k, v) => mk.append(el('div', { class: 'k' }, [k]), el('div', { class: 'num' }, [String(v)]));
    add('corner error mean', `${fmt(m.corner_err_mean_px)} px`);
    add('corner error p95', `${fmt(m.corner_err_p95_px)} px`);
    add('corner error max', `${fmt(m.corner_err_max_px)} px`);
    add('scale rel. error', fmt(m.scale_rel_err_mean, 4));
    add('rotation error', `${fmt(m.rotation_err_deg_mean)}°`);
    add('perspective error', fmt(m.perspective_err_mean, 6));
    add('corner spread', `${fmt(m.corner_spread_mean_px)} px`);
    add('occlusion error', fmt(m.occlusion_err_mean, 3));
    add('overlay slip (mean)', `${fmt(ev.slip && ev.slip.slip_px_mean)} px`);
    add('overlay slip (p95)', `${fmt(ev.slip && ev.slip.slip_px_p95)} px`);
    add('honesty cases', `${m.honest_cases} / ${m.honesty_cases}`);
    add('grade', ev.metrics.fidelity ? `${ev.metrics.fidelity.grade} (${ev.metrics.fidelity.score})` : 'n/a');
    panel.append(mk);
    if (ev.slip) panel.append(el('div', { class: 'hint' }, [`Overlay slip compares the composited content against the true plane over ${ev.slip.samples} sampled frames.`]));
    const dec = ev.decisions;
    if (dec) {
      panel.append(el('h3', { style: 'margin-top:12px' }, ['Decision audit']));
      const dk = el('div', { class: 'kv' });
      const addD = (k, v) => dk.append(el('div', { class: 'k' }, [k]), el('div', { class: 'num' }, [String(v)]));
      addD('non-tracking frames', dec.non_tracking_frames);
      for (const [state, n] of Object.entries(dec.by_state || {})) addD(`  ${state}`, n);
      panel.append(dk);
      panel.append(el('div', { class: 'hint' }, [
        'Predicted/paused frames are listed against the true occlusion so you can confirm the '
        + 'decision came from the observation. Scrub them in Review & evidence.']));
    }
  }
  ws.append(panel);
  if (STATE.job && STATE.truth) ws.append(el('div', { class: 'footer' }, ['Manual anchor saved separately from tracking results in the job folder (job.json vs track.json).']));
}

/* ---------------- boot ---------------- */

document.addEventListener('DOMContentLoaded', async () => {
  $('#btn-generate').addEventListener('click', async () => {
    await api('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    toast('Clip generation started (existing clips will be replaced when done)');
  });
  await loadClips();
  await loadJobs();
  if (STATE.jobs.length) openJob(STATE.jobs[0].id);
  window.addEventListener('keydown', (e) => {
    if (STATE.tab !== 'review') return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  });
});
