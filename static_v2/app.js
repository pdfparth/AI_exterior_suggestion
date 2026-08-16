/* ============================================================================
   Renovation Studio v2 — application logic
   Vanilla JS, no build step. Same API as v1; this is purely a new front end.
   ========================================================================== */

const S = {
  projectId: null,
  analysis: null,
  catalog: null,
  selections: {},       // regionId -> materialId
  enabled: {},          // regionId -> bool
  rateOverrides: {},    // materialId -> {material_rate, labour_rate}
  scaleOverride: null,
  estimate: null,
  engine: 'auto',
  step: 1,
  ratesOpen: false,
};

const $  = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const icon = (name, size = 16) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none"><use href="#i-${name}"/></svg>`;

// Distinct hue per component, shared by the SVG overlay and the region list so
// the eye can connect a row to a shape on the photo.
const HUE = {
  wall: '#5b8def', window: '#f5b544', door: '#ff6b6b', balcony: '#a878f0',
  pillar: '#2dd4a7', parapet: '#e8c44a', railing: '#4fc3e8', gate: '#d98a4a',
  garage: '#9b7ede', stairs: '#4dd6c1', roof_edge: '#8d9bb5',
};
const hue = (l) => HUE[l] || '#7b879e';

const money = (v) => '₹' + Math.round(v).toLocaleString('en-IN');
const title = (s) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

/* ---------- toasts ------------------------------------------------------ */

const TOAST_ICON = { ok: 'i-ok', err: 'i-alert', warn: 'i-alert', info: 'i-info' };

function toast(kind, title_, body, ms = 5200) {
  const t = el('div', `toast ${kind}`);
  t.innerHTML = `
    <svg class="ic" viewBox="0 0 24 24" fill="none"><use href="#${TOAST_ICON[kind]}"/></svg>
    <div class="tx">${title_ ? `<b>${title_}</b>` : ''}${body || ''}</div>`;
  $('toasts').appendChild(t);
  const kill = () => {
    t.classList.add('out');
    t.addEventListener('animationend', () => t.remove(), { once: true });
  };
  if (ms) setTimeout(kill, ms);
  t.onclick = kill;
  return t;
}

/* ---------- api --------------------------------------------------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let d = `HTTP ${res.status}`;
    try { d = (await res.json()).detail || d; } catch {}
    const e = new Error(d);
    e.status = res.status;
    throw e;
  }
  return res.json();
}

/* ---------- stepper ----------------------------------------------------- */

function setStep(n) {
  S.step = Math.max(S.step, n);
  document.querySelectorAll('.step').forEach((s) => {
    const i = +s.dataset.step;
    s.classList.toggle('done', i < S.step);
    s.classList.toggle('active', i === S.step);
    const dot = s.querySelector('.step-dot');
    dot.innerHTML = i < S.step
      ? `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="3"><use href="#i-check"/></svg>`
      : i;
  });
}

function reveal(id) {
  const p = $(id);
  if (!p.hidden) return p;
  p.hidden = false;
  p.classList.add('rise');
  return p;
}

function scrollTo(id) {
  requestAnimationFrame(() =>
    $(id).scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

/* ---------- health ------------------------------------------------------ */

(async () => {
  try {
    const h = await api('/api/health');
    const c = $('health');
    if (h.gemini_key_present) {
      c.className = 'chip live';
      c.innerHTML = `<span class="dot"></span><span>${h.vision_model}</span>`;
    } else {
      c.className = 'chip down';
      c.innerHTML = `<span class="dot"></span><span>No API key</span>`;
      toast('err', 'Gemini API key missing',
        'Create a <code>.env</code> file with <code>GEMINI_API_KEY=…</code> and restart the server.', 0);
    }
  } catch {
    $('health').className = 'chip down';
    $('health').innerHTML = `<span class="dot"></span><span>Offline</span>`;
  }
})();

/* ---------- 1. upload --------------------------------------------------- */

const drop = $('drop');
drop.onclick = () => $('file').click();
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
};
$('file').onchange = (e) => e.target.files[0] && upload(e.target.files[0]);

async function upload(file) {
  const host = $('upload-msg');
  host.innerHTML = '';
  const work = el('div', 'work pop', `
    <div class="ring"></div>
    <div><div class="txt">Checking your photo…</div>
    <div class="sub">${file.name} · ${(file.size / 1048576).toFixed(1)} MB</div></div>`);
  host.appendChild(work);

  // Show the picked image immediately - the upload feels instant even though
  // the quality gate is a network round trip.
  const localUrl = URL.createObjectURL(file);
  $('orig-img').src = localUrl;
  $('cmp-before').src = localUrl;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await api('/api/projects', { method: 'POST', body: fd });

    if (!r.accepted) {
      host.innerHTML = '';
      host.appendChild(el('div', 'note err', `
        ${icon('alert')}
        <div><b>This photo can't be used.</b><br>${r.reason}
        ${r.guidance ? `<br><br><b>Try this:</b> ${r.guidance}` : ''}</div>`));
      toast('err', 'Photo rejected', r.guidance || r.reason);
      return;
    }

    S.projectId = r.project_id;
    host.innerHTML = '';
    $('btn-restart').hidden = false;
    const src = `/api/projects/${r.project_id}/image/original`;
    $('orig-img').src = src;
    $('cmp-before').src = src;
    toast('ok', 'Photo accepted', `${r.width}×${r.height}px · ${r.reason}`);
    await analyse();
  } catch (e) {
    host.innerHTML = '';
    host.appendChild(el('div', 'note err', `${icon('alert')}<div>${e.message}</div>`));
    toast('err', 'Upload failed', e.message, 8000);
  }
}

$('btn-restart').onclick = () => location.reload();

/* ---------- samples ----------------------------------------------------- */

fetch('/api/samples').then((r) => r.json()).then((list) => {
  if (!list?.length) return;
  $('samples').hidden = false;
  const grid = $('sample-grid');
  list.forEach((name, i) => {
    const card = el('div', 'sample');
    card.style.animationDelay = `${i * 60}ms`;
    card.classList.add('pop');
    card.innerHTML = `<img src="/api/samples/${encodeURIComponent(name)}" alt="${name}">`;
    card.onclick = async () => {
      const blob = await (await fetch(`/api/samples/${encodeURIComponent(name)}`)).blob();
      upload(new File([blob], name, { type: blob.type || 'image/jpeg' }));
    };
    grid.appendChild(card);
  });
}).catch(() => {});

/* ---------- 2. survey --------------------------------------------------- */

async function analyse() {
  setStep(2);
  reveal('p-survey');
  scrollTo('p-survey');

  const load = $('survey-loading');
  load.innerHTML = '';
  load.appendChild(el('div', 'work', `
    <div class="ring"></div>
    <div><div class="txt">Surveying the facade…</div>
    <div class="sub">Identifying surfaces and establishing real-world scale</div>
    <div class="bar"><i></i></div></div>`));

  try {
    const r = await api(`/api/projects/${S.projectId}/analyse`, { method: 'POST' });
    S.analysis = r.analysis;
    S.catalog = r.catalog;
    S.selections = r.selections || {};
    S.enabled = {};
    S.analysis.regions.forEach((x) => { S.enabled[x.id] = true; });

    load.innerHTML = '';
    $('survey-body').hidden = false;
    $('survey-count').hidden = false;
    $('survey-count').innerHTML =
      `${icon('layers', 13)} ${S.analysis.regions.length} surfaces · ${S.analysis.storeys} storey(s)`;

    renderScale();
    renderRegions();
    renderOverlay();
    renderSurveyWarnings();

    setStep(3);
    reveal('p-design');
    renderMaterials();
    reveal('p-visual');
    await recalc();
    toast('ok', 'Survey complete',
      `${S.analysis.regions.length} surfaces identified. ${S.analysis.style_note || ''}`);
  } catch (e) {
    load.innerHTML = '';
    load.appendChild(el('div', 'note err', `${icon('alert')}<div>${e.message}</div>`));
    toast('err', 'Survey failed', e.message, 9000);
  }
}

function renderScale() {
  const s = S.analysis.scale;
  $('scale-card').innerHTML = `
    <h4>Real-world scale</h4>
    <div class="why">
      Measured against <b>${s.reference_object}</b> (assumed ${s.reference_real_feet.toFixed(2)} ft).
      Confidence ${(s.confidence * 100).toFixed(0)}%.
    </div>
    <div class="dims">
      <div class="dim">
        <label>Facade width</label>
        <div class="dim-input"><input id="sc-w" type="number" step="1"
          value="${s.building_width_ft.toFixed(0)}"><span>ft</span></div>
      </div>
      <div class="dim">
        <label>Facade height</label>
        <div class="dim-input"><input id="sc-h" type="number" step="1"
          value="${s.building_height_ft.toFixed(0)}"><span>ft</span></div>
      </div>
    </div>
    <div class="why" style="margin-top:10px;font-size:11.5px">
      Every area scales from these two numbers — correct them if they look wrong.
    </div>`;

  ['sc-w', 'sc-h'].forEach((id) => {
    $(id).onchange = () => {
      S.scaleOverride = {
        building_width_ft: parseFloat($('sc-w').value) || s.building_width_ft,
        building_height_ft: parseFloat($('sc-h').value) || s.building_height_ft,
      };
      updateRail();
      recalc();
      toast('info', 'Scale updated', 'All areas and costs have been recalculated.', 3000);
    };
  });
}

function polyArea(p) {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

function renderRegions() {
  const host = $('regions');
  host.innerHTML = '';
  host.classList.add('stagger');

  [...S.analysis.regions]
    .sort((a, b) => polyArea(b.polygon) - polyArea(a.polygon))
    .forEach((r, i) => {
      const row = el('div', `region on`);
      row.dataset.rid = r.id;
      row.style.setProperty('--i', i);
      row.innerHTML = `
        <div class="tick"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><use href="#i-check"/></svg></div>
        <div class="bar" style="background:${hue(r.label)}"></div>
        <div>
          <div class="nm">${title(r.label)}</div>
          ${r.note ? `<div class="meta">${r.note}</div>` : ''}
        </div>
        <div class="pct">${(polyArea(r.polygon) * 100).toFixed(1)}%</div>`;

      row.onmouseenter = () => hot(r.id, true);
      row.onmouseleave = () => hot(r.id, false);
      row.onclick = () => toggleRegion(r.id);
      host.appendChild(row);
    });
}

const NS = 'http://www.w3.org/2000/svg';

function renderOverlay() {
  const svg = $('overlay');
  svg.innerHTML = '';

  // Largest first so small components paint above big ones.
  [...S.analysis.regions]
    .sort((a, b) => polyArea(b.polygon) - polyArea(a.polygon))
    .forEach((r, i) => {
      const pts = r.polygon.map(([x, y]) => `${x * 100},${y * 100}`).join(' ');

      // Perimeter in viewBox units drives the draw-on dash length.
      let per = 0;
      for (let k = 0; k < r.polygon.length; k++) {
        const [x1, y1] = r.polygon[k], [x2, y2] = r.polygon[(k + 1) % r.polygon.length];
        per += Math.hypot((x2 - x1) * 100, (y2 - y1) * 100);
      }

      const g = document.createElementNS(NS, 'g');
      g.dataset.id = r.id;
      g.style.setProperty('--len', per.toFixed(1));
      g.style.setProperty('--delay', `${i * 85}ms`);
      if (!S.enabled[r.id]) g.classList.add('off');

      const face = document.createElementNS(NS, 'polygon');
      face.setAttribute('points', pts);
      face.setAttribute('fill', hue(r.label));
      face.classList.add('face');

      const edge = document.createElementNS(NS, 'polygon');
      edge.setAttribute('points', pts);
      edge.setAttribute('stroke', hue(r.label));
      edge.setAttribute('vector-effect', 'non-scaling-stroke');
      edge.classList.add('edge');

      g.append(face, edge);
      g.onmouseenter = () => hot(r.id, true);
      g.onmouseleave = () => hot(r.id, false);
      g.onclick = () => toggleRegion(r.id);
      svg.appendChild(g);
    });
}

function hot(id, on) {
  const g = $('overlay').querySelector(`g[data-id="${id}"]`);
  if (g) g.classList.toggle('hot', on);
  const row = $('regions').querySelector(`[data-rid="${id}"]`);
  if (row) row.classList.toggle('peek', on);
}

/* Toggling is reachable from both the list and the photo, so it lives here. */
function toggleRegion(id) {
  S.enabled[id] = !S.enabled[id];
  const row = $('regions').querySelector(`[data-rid="${id}"]`);
  if (row) {
    row.classList.toggle('on', S.enabled[id]);
    row.classList.toggle('muted', !S.enabled[id]);
  }
  const g = $('overlay').querySelector(`g[data-id="${id}"]`);
  if (g) g.classList.toggle('off', !S.enabled[id]);
  renderMaterials();
  recalc();
}

function renderSurveyWarnings() {
  const host = $('survey-warn');
  host.innerHTML = '';
  const w = S.analysis.warnings || [];
  if (!w.length) return;
  host.appendChild(el('div', 'note warn', `
    ${icon('alert')}
    <div><b>Affects accuracy</b><ul style="margin:6px 0 0;padding-left:17px">
      ${w.map((x) => `<li style="margin-bottom:3px">${x}</li>`).join('')}
    </ul></div>`));
}

/* ---------- 3. materials ------------------------------------------------ */

function renderMaterials() {
  const host = $('materials');
  host.innerHTML = '';

  const byLabel = {};
  S.analysis.regions.forEach((r) => {
    if (S.enabled[r.id]) (byLabel[r.label] ||= []).push(r);
  });

  const mats = Object.fromEntries(S.catalog.materials.map((m) => [m.id, m]));
  const labels = Object.keys(byLabel);

  if (!labels.length) {
    host.appendChild(el('div', 'empty', `${icon('layers', 34)}<p>No surfaces selected.</p>`));
    return;
  }

  labels.forEach((label, li) => {
    const regions = byLabel[label];
    const opts = S.catalog.by_label[label] || [];
    if (!opts.length) return;

    const sec = el('div', 'mat-section rise');
    sec.style.animationDelay = `${li * 70}ms`;
    sec.appendChild(el('div', 'mat-head',
      `<h4>${title(label)}</h4><span class="pill">${regions.length} surface${regions.length > 1 ? 's' : ''}</span>`));

    const grid = el('div', 'mat-grid');

    // "Leave as is" removes this component from both the render and the cost.
    const none = el('div', 'mat');
    const anySel = regions.some((r) => S.selections[r.id]);
    if (!anySel) none.classList.add('on');
    none.innerHTML = `
      <div class="mat-sw" style="background:
        repeating-linear-gradient(45deg,#1b2130,#1b2130 7px,#232b3c 7px,#232b3c 14px)"></div>
      <div class="mat-body"><div class="mat-nm">Leave as is</div>
      <div class="mat-rate">Not renovated</div></div>
      <div class="mat-check">${icon('check', 12)}</div>`;
    none.onclick = () => {
      regions.forEach((r) => delete S.selections[r.id]);
      renderMaterials(); recalc();
    };
    grid.appendChild(none);

    opts.forEach((mid) => {
      const m = mats[mid];
      const card = el('div', 'mat');
      if (regions.every((r) => S.selections[r.id] === mid)) card.classList.add('on');
      const rate = m.linear ? `${money(m.material_rate)}/rft` : `${money(m.material_rate)}/${m.unit}`;
      card.innerHTML = `
        <div class="mat-sw" style="background:${m.swatch_css}"></div>
        <div class="mat-body"><div class="mat-nm">${m.name}</div>
        <div class="mat-rate">${rate}</div></div>
        <div class="mat-check">${icon('check', 12)}</div>`;
      card.onclick = () => {
        regions.forEach((r) => { S.selections[r.id] = mid; });
        renderMaterials(); recalc();
      };
      grid.appendChild(card);
    });

    sec.appendChild(grid);
    host.appendChild(sec);
  });
}

const activeSelections = () =>
  Object.fromEntries(Object.entries(S.selections).filter(([rid]) => S.enabled[rid]));

/* ---------- 4. visualise ------------------------------------------------ */

$('engine-seg').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  S.engine = b.dataset.engine;
  [...$('engine-seg').children].forEach((c) => c.classList.toggle('on', c === b));
};

$('btn-render').onclick = async () => {
  const sel = activeSelections();
  const host = $('render-msg');
  const btn = $('btn-render');
  host.innerHTML = '';

  if (!Object.keys(sel).length) {
    toast('warn', 'Nothing selected', 'Choose at least one material first.');
    return;
  }

  setStep(4);
  btn.disabled = true;
  btn.classList.add('busy');
  const label = S.engine === 'local' ? 'Compositing locally…' : 'Generating with Gemini…';
  host.appendChild(el('div', 'work pop', `
    <div class="ring"></div>
    <div><div class="txt">${label}</div>
    <div class="sub">${S.engine === 'local'
      ? 'Applying materials to the detected surfaces'
      : 'Photorealistic render — this usually takes 10–30 seconds'}</div>
    <div class="bar"><i></i></div></div>`));

  try {
    const r = await api(`/api/projects/${S.projectId}/design`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selections: sel, engine: S.engine }),
    });

    await showCompare(r.image_url + '?t=' + Date.now(), r.engine);
    host.innerHTML = '';

    if (r.engine === 'local' && S.engine === 'auto') {
      host.appendChild(el('div', 'note warn',
        `${icon('alert')}<div><b>Rendered locally.</b> ${r.note}</div>`));
      toast('warn', 'Gemini unavailable', 'Fell back to an instant local render.', 7000);
    } else {
      toast('ok', 'Redesign ready',
        r.engine === 'local' ? 'Composited locally.' : 'Generated with Gemini.');
    }
    setStep(5);
    scrollTo('p-visual');
  } catch (e) {
    host.innerHTML = '';
    host.appendChild(el('div', 'note err', `${icon('alert')}<div>${e.message}</div>`));
    toast('err', 'Could not render', e.message, 10000);
  } finally {
    btn.disabled = false;
    btn.classList.remove('busy');
  }
};

/* Reveal animation: the "after" image wipes across from the left once loaded,
   then settles at the midpoint so the handle is obviously draggable. */
function showCompare(url, engine) {
  return new Promise((resolve) => {
    const after = $('cmp-after');
    $('engine-label').textContent =
      engine === 'local' ? 'local composite preview' : 'AI-generated visualisation';

    after.onload = () => {
      $('compare-wrap').hidden = false;
      $('compare-wrap').classList.add('pop');
      const c = $('compare');

      // Animate 0% -> 100% -> 50%: a full reveal, then park in the middle.
      let t0 = null;
      const dur = 1500;
      const ease = (x) => 1 - Math.pow(1 - x, 3);
      const frame = (ts) => {
        if (!t0) t0 = ts;
        const p = Math.min(1, (ts - t0) / dur);
        // 0→100 over the first 62%, then 100→50 over the rest.
        const pos = p < .62
          ? ease(p / .62) * 100
          : 100 - ease((p - .62) / .38) * 50;
        setPos(pos, false);
        if (p < 1) requestAnimationFrame(frame);
        else resolve();
      };
      requestAnimationFrame(frame);
    };
    after.onerror = () => resolve();
    after.src = url;
  });
}

/* ---------- compare slider (pointer + keyboard) ------------------------- */

const cmp = $('compare');
let pos = 50, dragging = false;

function setPos(p, store = true) {
  p = Math.max(0, Math.min(100, p));
  if (store) pos = p;
  cmp.style.setProperty('--pos', p + '%');
  cmp.querySelector('.compare-handle').style.left = p + '%';
}

function posFromEvent(e) {
  const r = cmp.getBoundingClientRect();
  return ((e.clientX - r.left) / r.width) * 100;
}

cmp.addEventListener('pointerdown', (e) => {
  dragging = true;
  cmp.setPointerCapture(e.pointerId);
  setPos(posFromEvent(e));
});
cmp.addEventListener('pointermove', (e) => { if (dragging) setPos(posFromEvent(e)); });
cmp.addEventListener('pointerup',     (e) => { dragging = false; cmp.releasePointerCapture(e.pointerId); });
cmp.addEventListener('pointercancel', () => { dragging = false; });

// Hovering (without dragging) nudges the split - feels alive, stays subtle.
cmp.addEventListener('pointermove', (e) => {
  if (dragging || e.pointerType !== 'mouse') return;
  const p = posFromEvent(e);
  setPos(pos + (p - pos) * 0.12, false);
});
cmp.addEventListener('pointerleave', () => { if (!dragging) setPos(pos, false); });

cmp.tabIndex = 0;
cmp.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft')  { setPos(pos - 4); e.preventDefault(); }
  if (e.key === 'ArrowRight') { setPos(pos + 4); e.preventDefault(); }
});

/* ---------- 5. estimate ------------------------------------------------- */

async function recalc() {
  const sel = activeSelections();
  reveal('p-estimate');
  const host = $('estimate-body');

  if (!Object.keys(sel).length) {
    host.innerHTML = '';
    host.appendChild(el('div', 'empty', `${icon('doc', 34)}<p>Select materials to see the estimate.</p>`));
    $('rail-summary').hidden = true;
    return;
  }

  try {
    S.estimate = await api(`/api/projects/${S.projectId}/estimate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        selections: sel,
        rate_overrides: S.rateOverrides,
        scale_override: S.scaleOverride,
      }),
    });
    renderEstimate();
    updateRail();
  } catch (e) {
    host.innerHTML = '';
    host.appendChild(el('div', 'note err', `${icon('alert')}<div>${e.message}</div>`));
  }
}

/* Count up to the new total rather than snapping - makes a rate change legible. */
function countTo(node, to, from = 0, ms = 850) {
  const t0 = performance.now();
  const ease = (x) => 1 - Math.pow(1 - x, 3);
  const tick = (ts) => {
    const p = Math.min(1, (ts - t0) / ms);
    node.textContent = money(from + (to - from) * ease(p));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderEstimate() {
  const e = S.estimate;
  const host = $('estimate-body');
  const prev = host.dataset.total ? +host.dataset.total : 0;
  host.innerHTML = '';
  host.dataset.total = e.grand_total;

  // --- hero total ---
  const matPct = e.grand_total ? (e.material_total / e.grand_total) * 100 : 50;
  const hero = el('div', 'total-hero pop', `
    <div class="total-lbl">Estimated total</div>
    <div class="total-val" id="hero-total">${money(e.grand_total)}</div>
    <div class="total-split">
      <div class="split-item"><div class="k">Material</div><div class="v">${money(e.material_total)}</div></div>
      <div class="split-item"><div class="k">Labour</div><div class="v">${money(e.labour_total)}</div></div>
      <div class="split-item"><div class="k">Line items</div><div class="v">${e.line_items.length}</div></div>
    </div>
    <div class="split-bar">
      <i style="width:${matPct}%;background:linear-gradient(90deg,var(--gold),#dd9b28)"></i>
      <i style="width:${100 - matPct}%;background:var(--surface-3)"></i>
    </div>`);
  host.appendChild(hero);
  countTo($('hero-total', hero) || hero.querySelector('#hero-total'), e.grand_total, prev);

  if (!e.line_items.length) return;

  // --- rate editor (5.7). The full breakdown lives in the PDF. ---
  const seen = new Set();
  const rows = e.line_items.filter((i) => !seen.has(i.material_id) && seen.add(i.material_id));

  const det = el('details', 'rates');
  // Survives the re-render that follows every rate edit; without this the
  // panel snaps shut the moment you change a number.
  det.open = S.ratesOpen === true;
  det.addEventListener('toggle', () => { S.ratesOpen = det.open; });
  det.innerHTML = `
    <summary>Adjust rates <span class="pill">${rows.length}</span>
      <span class="chev">${icon('chev', 14)}</span></summary>
    <table>
      <thead><tr><th>Material</th><th class="n">Material rate</th><th class="n">Labour rate</th></tr></thead>
      <tbody>${rows.map((i) => `
        <tr>
          <td>${i.material_name}</td>
          <td class="n"><input class="rate-in" type="number" step="1" value="${i.material_rate}"
              data-mat="${i.material_id}" data-kind="material_rate"><span class="unit">/${i.material_unit_label}</span></td>
          <td class="n"><input class="rate-in" type="number" step="1" value="${i.labour_rate}"
              data-mat="${i.material_id}" data-kind="labour_rate"><span class="unit">/${i.unit}</span></td>
        </tr>`).join('')}</tbody>
    </table>`;
  host.appendChild(det);

  det.querySelectorAll('.rate-in').forEach((inp) => {
    inp.onchange = () => {
      (S.rateOverrides[inp.dataset.mat] ||= {})[inp.dataset.kind] = parseFloat(inp.value) || 0;
      S.ratesOpen = true;
      recalc();
    };
  });

  // --- download ---
  const dl = el('div', null, '');
  dl.style.cssText = 'margin-top:22px;display:flex;gap:12px;align-items:center;flex-wrap:wrap';
  const btn = el('button', 'btn gold', `${icon('down')} Download PDF report`);
  btn.onclick = () => {
    window.location = `/api/projects/${S.projectId}/report`;
    toast('ok', 'Report downloading', 'Full breakdown, assumptions and working included.');
  };
  dl.appendChild(btn);
  dl.appendChild(el('span', null,
    `<span style="font-size:12.5px;color:var(--ink-4)">Areas, quantities, wastage and assumptions are in the PDF.</span>`));
  host.appendChild(dl);
}

function updateRail() {
  const e = S.estimate;
  if (!e) return;
  $('rail-summary').hidden = false;
  $('rail-total').textContent = money(e.grand_total);
  $('rail-mat').textContent = money(e.material_total);
  $('rail-lab').textContent = money(e.labour_total);
  const s = e.scale_used;
  $('rail-dims').textContent =
    `${s.building_width_ft.toFixed(0)} × ${s.building_height_ft.toFixed(0)} ft`;
}
