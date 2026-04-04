"""Generate an interactive HTML policy explorer from a trained QRN model.

Exports the GNN weights into a self-contained HTML file that runs the
forward pass in JavaScript, letting the user sweep observation features
via sliders and see live Q-value bar charts.

Usage:
    python diagnostics/policy_probes/generate_policy_explorer.py \
        --model checkpoints/cluster_004/policy.pth \
        --output checkpoints/cluster_004/policy_explorer.html
"""

from __future__ import annotations
import argparse, json, os
import torch
from rl_stack.model import QNetwork
from rl_stack.env_wrapper import N_ACTIONS


def _export_weights(model_path: str, hidden: int = 64) -> dict:
    model = QNetwork(node_dim=8, hidden=hidden, n_actions=N_ACTIONS)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    weights = {}
    for name, param in model.state_dict().items():
        t = param.detach().float()
        if t.dim() == 2:
            weights[name] = [[round(v.item(), 6) for v in row] for row in t]
        else:
            weights[name] = [round(v.item(), 6) for v in t]
    return weights


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QRN Policy Explorer</title>
<style>
  :root {
    --bg: #1a1b26; --surface: #24283b; --border: #414868;
    --text: #c0caf5; --dim: #565f89; --accent: #7aa2f7;
    --wait-color: #9aa5ce; --swap-color: #f7768e; --purify-color: #9ece6a;
    --best-glow: #e0af68;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg); color: var(--text);
    display: flex; justify-content: center; padding: 24px;
    min-height: 100vh;
  }
  .container {
    display: grid;
    grid-template-columns: 340px 1fr;
    grid-template-rows: auto 1fr auto;
    gap: 20px;
    max-width: 900px; width: 100%;
  }
  .header {
    grid-column: 1 / -1;
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .subtitle { font-size: 12px; color: var(--dim); }

  /* Controls panel */
  .controls {
    background: var(--surface); border-radius: 12px;
    padding: 20px; border: 1px solid var(--border);
  }
  .control-group { margin-bottom: 16px; }
  .control-group:last-child { margin-bottom: 0; }
  .control-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; color: var(--dim); margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .control-value {
    font-size: 13px; color: var(--accent); font-weight: 600;
  }
  input[type="range"] {
    -webkit-appearance: none; width: 100%; height: 6px;
    border-radius: 3px; background: var(--border); outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: var(--accent); cursor: pointer;
    border: 2px solid var(--bg);
  }

  /* Toggle switch */
  .toggle-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0;
  }
  .toggle-label { font-size: 12px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .toggle {
    position: relative; width: 44px; height: 24px; cursor: pointer;
  }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider {
    position: absolute; inset: 0; background: var(--border);
    border-radius: 12px; transition: 0.2s;
  }
  .toggle .slider::before {
    content: ''; position: absolute; width: 18px; height: 18px;
    left: 3px; bottom: 3px; background: var(--text);
    border-radius: 50%; transition: 0.2s;
  }
  .toggle input:checked + .slider { background: var(--accent); }
  .toggle input:checked + .slider::before { transform: translateX(20px); }

  /* Q-value display */
  .qvalues {
    background: var(--surface); border-radius: 12px;
    padding: 24px; border: 1px solid var(--border);
    display: flex; flex-direction: column; justify-content: center;
  }
  .qvalues h2 {
    font-size: 14px; color: var(--dim); margin-bottom: 20px;
    text-transform: uppercase; letter-spacing: 1px;
  }
  .bar-row {
    display: flex; align-items: center; margin-bottom: 14px; gap: 12px;
  }
  .bar-label {
    width: 60px; font-size: 13px; font-weight: 600; text-align: right;
  }
  .bar-track {
    flex: 1; height: 32px; background: var(--bg); border-radius: 6px;
    overflow: hidden; position: relative;
  }
  .bar-fill {
    height: 100%; border-radius: 6px; transition: width 0.15s ease-out;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 8px; font-size: 11px; font-weight: 600;
    min-width: 50px;
  }
  .bar-fill.best {
    box-shadow: 0 0 12px var(--best-glow);
  }
  .bar-fill.wait   { background: var(--wait-color); color: var(--bg); }
  .bar-fill.swap   { background: var(--swap-color); color: var(--bg); }
  .bar-fill.purify { background: var(--purify-color); color: var(--bg); }

  .advantage-box {
    margin-top: 16px; padding: 12px; background: var(--bg);
    border-radius: 8px; font-size: 12px; line-height: 1.8;
  }
  .advantage-box .best-label {
    color: var(--best-glow); font-weight: 700; font-size: 14px;
  }

  /* Chain diagram */
  .chain-diagram {
    grid-column: 1 / -1;
    background: var(--surface); border-radius: 12px;
    padding: 16px; border: 1px solid var(--border);
    display: flex; flex-direction: column; align-items: center; gap: 8px;
  }
  .chain-diagram .label { font-size: 11px; color: var(--dim); }
  .chain-svg { width: 100%; max-width: 500px; height: 60px; }
  .node-circle {
    cursor: pointer; transition: all 0.15s;
  }
  .node-circle:hover { filter: brightness(1.3); }
  .node-text { font-size: 11px; fill: var(--bg); font-weight: 700; pointer-events: none; }
  .edge-line { stroke: var(--border); stroke-width: 2; }

  /* Presets */
  .presets {
    display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px;
  }
  .preset-btn {
    font-size: 10px; padding: 4px 10px; border-radius: 4px;
    background: var(--border); color: var(--text); border: none;
    cursor: pointer; font-family: inherit; text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .preset-btn:hover { background: var(--accent); color: var(--bg); }
  .preset-btn.active { background: var(--accent); color: var(--bg); }

  .section-title {
    font-size: 11px; color: var(--dim); margin-bottom: 10px;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid var(--border); padding-bottom: 4px;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>QRN Policy Explorer</h1>
    <span class="subtitle">Interactive Q-value viewer for trained quantum repeater agent</span>
  </div>

  <div class="controls">
    <div class="section-title">Presets</div>
    <div class="presets">
      <button class="preset-btn" onclick="applyPreset('highFid')">High Fidelity</button>
      <button class="preset-btn" onclick="applyPreset('lowFid')">Low Fidelity</button>
      <button class="preset-btn" onclick="applyPreset('earlyGame')">Early Game</button>
      <button class="preset-btn" onclick="applyPreset('lateGame')">Late Game</button>
      <button class="preset-btn" onclick="applyPreset('swapReady')">Swap Ready</button>
      <button class="preset-btn" onclick="applyPreset('purifyReady')">Purify Ready</button>
    </div>

    <div class="section-title">Probe Node Features</div>

    <div class="control-group">
      <div class="control-label">
        <span>Frac Occupied</span>
        <span class="control-value" id="val-occ">0.50</span>
      </div>
      <input type="range" id="sl-occ" min="0" max="1" step="0.25" value="0.5" oninput="update()">
    </div>

    <div class="control-group">
      <div class="control-label">
        <span>Mean Fidelity</span>
        <span class="control-value" id="val-fid">0.75</span>
      </div>
      <input type="range" id="sl-fid" min="0.25" max="1.0" step="0.01" value="0.75" oninput="update()">
    </div>

    <div class="toggle-row">
      <span class="toggle-label">Can Swap</span>
      <label class="toggle">
        <input type="checkbox" id="tgl-swap" checked onchange="update()">
        <span class="slider"></span>
      </label>
    </div>

    <div class="toggle-row">
      <span class="toggle-label">Can Purify</span>
      <label class="toggle">
        <input type="checkbox" id="tgl-purify" onchange="update()">
        <span class="slider"></span>
      </label>
    </div>

    <div class="control-group" style="margin-top: 12px;">
      <div class="control-label">
        <span>Time Remaining</span>
        <span class="control-value" id="val-time">0.50</span>
      </div>
      <input type="range" id="sl-time" min="0.0" max="1.0" step="0.01" value="0.5" oninput="update()">
    </div>
  </div>

  <div class="qvalues">
    <h2>Q-Values at Probe Node</h2>
    <div class="bar-row">
      <span class="bar-label" style="color:var(--wait-color)">Wait</span>
      <div class="bar-track">
        <div class="bar-fill wait" id="bar-wait" style="width:50%">0.000</div>
      </div>
    </div>
    <div class="bar-row">
      <span class="bar-label" style="color:var(--swap-color)">Swap</span>
      <div class="bar-track">
        <div class="bar-fill swap" id="bar-swap" style="width:50%">0.000</div>
      </div>
    </div>
    <div class="bar-row">
      <span class="bar-label" style="color:var(--purify-color)">Purify</span>
      <div class="bar-track">
        <div class="bar-fill purify" id="bar-purify" style="width:50%">0.000</div>
      </div>
    </div>
    <div class="advantage-box" id="advantage-box"></div>
  </div>

  <div class="chain-diagram">
    <div class="label">5-Node Chain (click to select probe node)</div>
    <svg class="chain-svg" viewBox="0 0 500 60">
      <line class="edge-line" x1="70" y1="30" x2="170" y2="30"/>
      <line class="edge-line" x1="170" y1="30" x2="270" y2="30"/>
      <line class="edge-line" x1="270" y1="30" x2="370" y2="30"/>
      <line class="edge-line" x1="370" y1="30" x2="470" y2="30"/>
      <circle class="node-circle" id="node-0" cx="70"  cy="30" r="18" fill="var(--dim)" onclick="selectNode(0)"/>
      <circle class="node-circle" id="node-1" cx="170" cy="30" r="18" fill="var(--border)" onclick="selectNode(1)"/>
      <circle class="node-circle" id="node-2" cx="270" cy="30" r="18" fill="var(--accent)" onclick="selectNode(2)"/>
      <circle class="node-circle" id="node-3" cx="370" cy="30" r="18" fill="var(--border)" onclick="selectNode(3)"/>
      <circle class="node-circle" id="node-4" cx="470" cy="30" r="18" fill="var(--dim)" onclick="selectNode(4)"/>
      <text class="node-text" x="70"  y="34" text-anchor="middle">S</text>
      <text class="node-text" x="170" y="34" text-anchor="middle">1</text>
      <text class="node-text" x="270" y="34" text-anchor="middle">2</text>
      <text class="node-text" x="370" y="34" text-anchor="middle">3</text>
      <text class="node-text" x="470" y="34" text-anchor="middle">D</text>
    </svg>
  </div>
</div>

<script>
// ---- Model weights (embedded) ----
const W = __WEIGHTS_JSON__;

// ---- GNN forward pass ----
const NEIGHBORS = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]};
const ACTION_NAMES = ['Wait', 'Swap', 'Purify'];

function linear(x, weight, bias) {
  const out = new Array(weight.length);
  for (let o = 0; o < weight.length; o++) {
    let s = bias ? bias[o] : 0;
    const row = weight[o];
    for (let i = 0; i < x.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function relu(x) {
  return x.map(v => v > 0 ? v : 0);
}

function sageConv(X, lw, lb, rw) {
  const N = X.length, D = lw.length;
  const out = [];
  for (let i = 0; i < N; i++) {
    const nbrs = NEIGHBORS[i];
    const aggr = new Array(X[0].length).fill(0);
    for (const j of nbrs)
      for (let d = 0; d < X[0].length; d++) aggr[d] += X[j][d];
    const cnt = nbrs.length || 1;
    for (let d = 0; d < aggr.length; d++) aggr[d] /= cnt;
    const left = linear(aggr, lw, lb);
    const right = linear(X[i], rw, null);
    const res = new Array(D);
    for (let d = 0; d < D; d++) res[d] = left[d] + right[d];
    out.push(res);
  }
  return out;
}

function forward(features) {
  let X = features.map(row => [...row]);

  X = sageConv(X,
    W['conv1.lin_l.weight'], W['conv1.lin_l.bias'], W['conv1.lin_r.weight']);
  X = X.map(relu);

  X = sageConv(X,
    W['conv2.lin_l.weight'], W['conv2.lin_l.bias'], W['conv2.lin_r.weight']);
  X = X.map(relu);

  X = sageConv(X,
    W['conv3.lin_l.weight'], W['conv3.lin_l.bias'], W['conv3.lin_r.weight']);
  X = X.map(relu);

  const qvals = [];
  for (let i = 0; i < X.length; i++) {
    let h = linear(X[i], W['head.0.weight'], W['head.0.bias']);
    h = relu(h);
    h = linear(h, W['head.2.weight'], W['head.2.bias']);
    qvals.push(h);
  }
  return qvals;
}

// ---- UI State ----
let probeNode = 2;

function getContextFeatures(tRem) {
  return [
    [0.25, 0.70, 1, 0, 0.25, 0, 0, tRem],  // node 0 (source)
    [0.50, 0.70, 0, 0, 0.50, 1, 0, tRem],  // node 1 (interior)
    [0.50, 0.70, 0, 0, 0.50, 1, 0, tRem],  // node 2 (interior)
    [0.50, 0.70, 0, 0, 0.50, 1, 0, tRem],  // node 3 (interior)
    [0.25, 0.70, 0, 1, 0.25, 0, 0, tRem],  // node 4 (dest)
  ];
}

function readSliders() {
  return {
    occ:    parseFloat(document.getElementById('sl-occ').value),
    fid:    parseFloat(document.getElementById('sl-fid').value),
    swap:   document.getElementById('tgl-swap').checked ? 1 : 0,
    purify: document.getElementById('tgl-purify').checked ? 1 : 0,
    time:   parseFloat(document.getElementById('sl-time').value),
  };
}

function update() {
  const s = readSliders();

  document.getElementById('val-occ').textContent   = s.occ.toFixed(2);
  document.getElementById('val-fid').textContent   = s.fid.toFixed(2);
  document.getElementById('val-time').textContent  = s.time.toFixed(2);

  const avail = 1.0 - s.occ;
  const ctx = getContextFeatures(s.time);
  const features = [];
  for (let i = 0; i < 5; i++) {
    if (i === probeNode) {
      features.push([s.occ, s.fid, 0, 0, avail, s.swap, s.purify, s.time]);
    } else {
      features.push([...ctx[i]]);
    }
  }

  const qvals = forward(features);
  const q = qvals[probeNode];

  const qMin = Math.min(...q);
  const qMax = Math.max(...q);
  const range = qMax - qMin || 1e-6;
  const bestIdx = q.indexOf(qMax);

  const ids   = ['bar-wait', 'bar-swap', 'bar-purify'];
  const names = ['Wait', 'Swap', 'Purify'];

  for (let a = 0; a < 3; a++) {
    const bar = document.getElementById(ids[a]);
    const pct = 15 + 85 * (q[a] - qMin) / range;
    bar.style.width = pct + '%';
    bar.textContent = q[a].toFixed(4);
    bar.classList.toggle('best', a === bestIdx);
  }

  const advBox = document.getElementById('advantage-box');
  const bestName = names[bestIdx];
  const lines = [`<span class="best-label">Best: ${bestName}</span>`];
  for (let a = 0; a < 3; a++) {
    if (a === bestIdx) continue;
    const diff = q[bestIdx] - q[a];
    lines.push(`${bestName} > ${names[a]} by <span style="color:var(--accent)">+${diff.toFixed(4)}</span>`);
  }
  advBox.innerHTML = lines.join('<br>');
}

function selectNode(n) {
  if (n === 0 || n === 4) return;
  probeNode = n;
  const colors = ['var(--dim)', 'var(--border)', 'var(--border)', 'var(--border)', 'var(--dim)'];
  colors[n] = 'var(--accent)';
  for (let i = 0; i < 5; i++) {
    document.getElementById('node-' + i).setAttribute('fill', colors[i]);
  }
  update();
}

const PRESETS = {
  highFid:     { occ: 0.5,  fid: 0.95, swap: true,  purify: false, time: 0.5 },
  lowFid:      { occ: 0.5,  fid: 0.35, swap: true,  purify: false, time: 0.5 },
  earlyGame:   { occ: 0.25, fid: 0.80, swap: false, purify: false, time: 0.9 },
  lateGame:    { occ: 0.75, fid: 0.60, swap: true,  purify: true,  time: 0.1 },
  swapReady:   { occ: 0.5,  fid: 0.75, swap: true,  purify: false, time: 0.5 },
  purifyReady: { occ: 0.5,  fid: 0.60, swap: false, purify: true,  time: 0.5 },
};

function applyPreset(name) {
  const p = PRESETS[name];
  document.getElementById('sl-occ').value   = p.occ;
  document.getElementById('sl-fid').value   = p.fid;
  document.getElementById('tgl-swap').checked   = p.swap;
  document.getElementById('tgl-purify').checked = p.purify;
  document.getElementById('sl-time').value  = p.time;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  update();
}

// Initial render
update();
</script>
</body>
</html>"""


def generate(model_path: str, output_path: str, hidden: int = 64) -> None:
    weights = _export_weights(model_path, hidden)
    weights_json = json.dumps(weights, separators=(',', ':'))
    html = _HTML_TEMPLATE.replace('__WEIGHTS_JSON__', weights_json)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML policy explorer")
    parser.add_argument("--model", default='checkpoints/cluster_004/policy.pth')
    parser.add_argument("--output", default='checkpoints/cluster_004/policy_explorer.html',
                        help="Output HTML path (default: same dir as model)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden dimension of the QNetwork")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.model), "policy_explorer.html")

    generate(args.model, args.output, args.hidden)
