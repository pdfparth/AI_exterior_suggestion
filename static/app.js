/* Frontend. Vanilla JS, no build step - open the page and it runs.
   State is one object; every render function reads from it. */

const S = {
  projectId: null,
  analysis: null,
  catalog: null,
  selections: {},      // regionId -> materialId
  enabled: {},         // regionId -> bool (unticked = excluded from costing)
  rateOverrides: {},   // materialId -> {material_rate, labour_rate}
  estimate: null,
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};

// Consistent colour per component type, shared between the SVG overlay and
// the region list so the eye can connect them.
const LABEL_COLOR = {
  wall: '#4d7cc7', window: '#e0932f', door: '#c7564d', balcony: '#8a5fc0',
  pillar: '#3f9e78', parapet: '#c9a227', railing: '#5aa9d6', gate: '#a56b3f',
  garage: '#8b6fb8', stairs: '#5f9ea0', roof_edge: '#7d8a99',
};
const colorFor = (l) => LABEL_COLOR[l] || '#888';
const rupees = (v) => 'Rs ' + Math.round(v).toLocaleString('en-IN');
const titleCase = (s) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

function msg(host, kind, text) {
  const box = el('div', `msg ${kind}`);
  if (kind === 'busy') box.innerHTML = `<span class="spinner"></span>${text}`;
  else box.innerHTML = text;
  host.innerHTML = '';
  host.appendChild(box);
  return box;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// --- health -----------------------------------------------------------

(async () => {
  try {
    const h = await api('/api/health');
    $('health').innerHTML = h.gemini_key_present
      ? `<span class="muted">Gemini connected &middot; ${h.vision_model} &middot; ${h.image_model}</span>`
      : `<span class="msg err" style="display:inline-block;padding:6px 10px">
           No <code>GEMINI_API_KEY</code> found. Create a <code>.env</code> file with
           <code>GEMINI_API_KEY=your_key</code> and restart the server.</span>`;
  } catch {}
})();

// --- stage 1: upload --------------------------------------------------

const drop = $('drop');
$('browse').onclick = (e) => { e.stopPropagation(); $('file').click(); };
drop.onclick = () => $('file').click();
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
};
$('file').onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); };

async function upload(file) {
  const status = $('upload-status');
  msg(status, 'busy', `Checking <strong>${file.name}</strong> is usable...`);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await api('/api/projects', { method: 'POST', body: fd });

    // 5.1: an unusable photo is rejected with guidance, not silently estimated.
    if (!r.accepted) {
      msg(status, 'err',
        `<strong>This photo can't be used.</strong><br>${r.reason}` +
        (r.guidance ? `<br><br><strong>Try this:</strong> ${r.guidance}` : ''));
      return;
    }

    S.projectId = r.project_id;
    msg(status, 'ok', `Photo accepted (${r.width}&times;${r.height}). ${r.reason}`);
    $('orig-img').src = `/api/projects/${r.project_id}/image/original`;
    $('cmp-before').src = `/api/projects/${r.project_id}/image/original`;
    await analyse();
  } catch (e) {
    msg(status, 'err', `Upload failed: ${e.message}`);
  }
}

// --- stage 2: analyse -------------------------------------------------

async function analyse() {
  $('stage-analyse').classList.remove('hidden');
  const host = $('region-list');
  msg(host, 'busy', 'Identifying surfaces and estimating scale...');
  $('stage-analyse').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const r = await api(`/api/projects/${S.projectId}/analyse`, { method: 'POST' });
    S.analysis = r.analysis;
    S.catalog = r.catalog;
    S.selections = r.selections;
    S.enabled = {};
    S.analysis.regions.forEach((x) => { S.enabled[x.id] = true; });

    renderScale();
    renderRegions();
    renderOverlay();
    renderWarnings();
    renderMaterials();
    $('stage-design').classList.remove('hidden');
    await recalc();
  } catch (e) {
    msg(host, 'err', `Analysis failed: ${e.message}`);
  }
}

function renderScale() {
  const s = S.analysis.scale;
  const box = $('scale-box');
  box.innerHTML = `
    <h3>Real-world scale</h3>
    <p class="tiny" style="margin:0 0 9px">
      Measured against <strong>${s.reference_object}</strong>
      (assumed ${s.reference_real_feet.toFixed(2)} ft).
      Confidence ${(s.confidence * 100).toFixed(0)}%.
    </p>
    <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
      <label class="tiny">Facade width
        <input class="rate-input" id="sc-w" type="number" step="1" value="${s.building_width_ft.toFixed(0)}"> ft
      </label>
      <label class="tiny">Facade height
        <input class="rate-input" id="sc-h" type="number" step="1" value="${s.building_height_ft.toFixed(0)}"> ft
      </label>
    </div>
    <p class="tiny muted" style="margin:9px 0 0">
      Every area scales from these two numbers &mdash; correct them if they look wrong.
    </p>`;

  // Editing scale must recost everything, so both inputs recalc on change.
  ['sc-w', 'sc-h'].forEach((id) => {
    $(id).onchange = () => {
      S.scaleOverride = {
        building_width_ft: parseFloat($('sc-w').value) || s.building_width_ft,
        building_height_ft: parseFloat($('sc-h').value) || s.building_height_ft,
      };
      recalc();
    };
  });
}

function renderRegions() {
  const host = $('region-list');
  host.innerHTML = '';
  host.appendChild(el('h3', 'tiny muted',
    `${S.analysis.regions.length} surfaces found &middot; ${S.analysis.storeys} storey(s)`));

  S.analysis.regions.forEach((r) => {
    const row = el('div', 'region');
    row.innerHTML = `
      <input type="checkbox" ${S.enabled[r.id] ? 'checked' : ''} data-id="${r.id}">
      <span class="swatch-dot" style="background:${colorFor(r.label)}"></span>
      <span class="name">${titleCase(r.label)}</span>
      <span class="note">${r.note || ''}</span>
      <span class="conf">${(r.confidence * 100).toFixed(0)}%</span>`;

    row.onmouseenter = () => highlight(r.id, true);
    row.onmouseleave = () => highlight(r.id, false);
    row.querySelector('input').onchange = (e) => {
      S.enabled[r.id] = e.target.checked;
      renderOverlay();
      renderMaterials();
      recalc();
    };
    host.appendChild(row);
  });
}

function renderOverlay() {
  const svg = $('overlay');
  svg.innerHTML = '';
  svg.setAttribute('viewBox', '0 0 100 100');

  // Largest first so small components (windows) sit on top of walls.
  const ordered = [...S.analysis.regions].sort((a, b) => polyArea(b.polygon) - polyArea(a.polygon));

  ordered.forEach((r) => {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    p.setAttribute('points', r.polygon.map(([x, y]) => `${x * 100},${y * 100}`).join(' '));
    p.setAttribute('fill', colorFor(r.label));
    p.setAttribute('stroke', colorFor(r.label));
    p.dataset.id = r.id;
    if (!S.enabled[r.id]) p.classList.add('off');
    svg.appendChild(p);
  });
}

function polyArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

function highlight(id, on) {
  const p = $('overlay').querySelector(`polygon[data-id="${id}"]`);
  if (p) p.classList.toggle('active', on);
}

function renderWarnings() {
  const host = $('analyse-warnings');
  host.innerHTML = '';
  const w = S.analysis.warnings || [];
  if (!w.length) return;
  msg(host, 'warn',
    `<strong>Things that affect accuracy</strong><ul>${w.map((x) => `<li>${x}</li>`).join('')}</ul>`);
}

// --- stage 3: materials -----------------------------------------------

function renderMaterials() {
  const host = $('material-picker');
  host.innerHTML = '';

  // Group by component type. Assigning per-label rather than per-polygon keeps
  // the UI manageable - nobody wants to dress six wall planes individually.
  const byLabel = {};
  S.analysis.regions.forEach((r) => {
    if (!S.enabled[r.id]) return;
    (byLabel[r.label] ||= []).push(r);
  });

  const mats = Object.fromEntries(S.catalog.materials.map((m) => [m.id, m]));

  Object.entries(byLabel).forEach(([label, regions]) => {
    const options = S.catalog.by_label[label] || [];
    if (!options.length) return;

    const group = el('div', 'mat-group');
    group.appendChild(el('h4', null,
      `${titleCase(label)} <span class="count">${regions.length} surface(s)</span>`));

    const chips = el('div', 'chips');

    // "None" excludes this component from the estimate entirely.
    const none = el('button', 'chip', 'Leave as is');
    const anySelected = regions.some((r) => S.selections[r.id]);
    if (!anySelected) none.classList.add('on');
    none.onclick = () => {
      regions.forEach((r) => delete S.selections[r.id]);
      renderMaterials(); recalc();
    };
    chips.appendChild(none);

    options.forEach((mid) => {
      const m = mats[mid];
      const chip = el('button', 'chip',
        `<span class="sw" style="background:${m.swatch_css}"></span>${m.name}`);
      if (regions.every((r) => S.selections[r.id] === mid)) chip.classList.add('on');
      chip.onclick = () => {
        regions.forEach((r) => { S.selections[r.id] = mid; });
        renderMaterials(); recalc();
      };
      chips.appendChild(chip);
    });

    group.appendChild(chips);
    host.appendChild(group);
  });
}

$('btn-render').onclick = async () => {
  const status = $('render-status');
  const btn = $('btn-render');
  const sel = activeSelections();

  if (!Object.keys(sel).length) {
    msg(status, 'warn', 'Pick at least one material first.');
    return;
  }

  const engine = document.querySelector('input[name="engine"]:checked')?.value || 'auto';
  btn.disabled = true;
  msg(status, 'busy', engine === 'local'
    ? 'Compositing locally...'
    : 'Generating the redesign with Gemini... this takes 10&ndash;30 seconds.');

  try {
    const r = await api(`/api/projects/${S.projectId}/design`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selections: sel, engine }),
    });
    // Cache-bust so a re-render actually shows the new image.
    $('cmp-after').src = r.image_url + '?t=' + Date.now();
    $('cmp-caption').textContent =
      r.engine === 'local' ? 'REDESIGNED — local preview' : 'REDESIGNED — Gemini';
    $('compare').classList.remove('hidden');

    // A local render after an attempted Gemini call means quota ran out; say
    // so plainly rather than passing off the cheaper render as the good one.
    if (r.engine === 'local' && engine === 'auto') {
      msg(status, 'warn', `<strong>Rendered locally.</strong> ${r.note}`);
    } else if (r.engine === 'local') {
      msg(status, 'ok', 'Composited locally.');
    } else {
      msg(status, 'ok', 'Redesign generated with Gemini.');
    }
  } catch (e) {
    msg(status, 'err', `Could not generate the redesign: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
};

// Only regions the user left ticked are costed or rendered.
function activeSelections() {
  return Object.fromEntries(
    Object.entries(S.selections).filter(([rid]) => S.enabled[rid])
  );
}

// --- stage 4: estimate ------------------------------------------------

async function recalc() {
  $('stage-estimate').classList.remove('hidden');
  const host = $('estimate-out');
  const sel = activeSelections();

  if (!Object.keys(sel).length) {
    msg(host, 'warn', 'No materials selected, so there is nothing to cost yet.');
    return;
  }

  try {
    S.estimate = await api(`/api/projects/${S.projectId}/estimate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        selections: sel,
        rate_overrides: S.rateOverrides,
        scale_override: S.scaleOverride || null,
      }),
    });
    renderEstimate();
  } catch (e) {
    msg(host, 'err', `Costing failed: ${e.message}`);
  }
}

function renderEstimate() {
  const e = S.estimate;
  const host = $('estimate-out');
  host.innerHTML = '';

  const head = el('div', 'headline');
  head.innerHTML = `
    <div><div class="lbl">Estimated total</div><div class="big">${rupees(e.grand_total)}</div></div>
    <div><div class="lbl">Material</div><div>${rupees(e.material_total)}</div></div>
    <div><div class="lbl">Labour</div><div>${rupees(e.labour_total)}</div></div>`;
  host.appendChild(head);

  if (!e.line_items.length) {
    host.appendChild(el('p', 'muted', 'Nothing selected to cost.'));
    return;
  }

  // The full breakdown - per-line areas, quantities, the working, assumptions
  // and warnings - lives in the PDF only. That is the document that goes to a
  // contractor; on screen the useful things are the total and the ability to
  // try a different rate.
  //
  // Rate editing stays because 5.7 requires it, but as a compact editor rather
  // than a column inside a breakdown table.
  const rates = el('details', 'rate-editor');
  const seen = new Set();
  const rows = e.line_items.filter((i) => {
    if (seen.has(i.material_id)) return false;
    seen.add(i.material_id);
    return true;
  });

  rates.innerHTML = `<summary>Adjust rates (${rows.length} material${
    rows.length === 1 ? '' : 's'})</summary>
    <table>
      <thead><tr><th>Material</th><th class="num">Material rate</th>
      <th class="num">Labour rate</th></tr></thead>
      <tbody>${rows.map((i) => `
        <tr>
          <td>${i.material_name}</td>
          <td class="num"><input class="rate-input" type="number" step="1"
                value="${i.material_rate}" data-mat="${i.material_id}"
                data-kind="material_rate"> /${i.material_unit_label}</td>
          <td class="num"><input class="rate-input" type="number" step="1"
                value="${i.labour_rate}" data-mat="${i.material_id}"
                data-kind="labour_rate"> /${i.unit}</td>
        </tr>`).join('')}
      </tbody></table>`;
  host.appendChild(rates);

  rates.querySelectorAll('.rate-input').forEach((inp) => {
    inp.onchange = () => {
      const mid = inp.dataset.mat;
      (S.rateOverrides[mid] ||= {})[inp.dataset.kind] = parseFloat(inp.value) || 0;
      recalc();
    };
  });

  host.appendChild(el('p', 'tiny muted',
    'The full breakdown &mdash; areas, quantities, wastage and assumptions &mdash; '
    + 'is in the PDF report.'));
}

$('btn-report').onclick = () => {
  window.location = `/api/projects/${S.projectId}/report`;
};

// --- sample images ----------------------------------------------------

fetch('/api/samples').then((r) => r.json()).then((list) => {
  if (!list.length) return;
  const host = $('samples');
  host.appendChild(el('span', 'muted tiny', 'or try a sample:'));
  list.forEach((name) => {
    const img = el('img');
    img.src = `/api/samples/${name}`;
    img.title = name;
    img.onclick = async (ev) => {
      ev.stopPropagation();
      const blob = await (await fetch(`/api/samples/${name}`)).blob();
      upload(new File([blob], name, { type: blob.type }));
    };
    host.appendChild(img);
  });
}).catch(() => {});
