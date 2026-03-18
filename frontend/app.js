/* ═══════════════════════════════════════════════════════════
   FraudShield AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000/api';

// ─── Global state ─────────────────────────────────────────
let dashStats   = { total: 0, fraud: 0, legit: 0, models: 0 };
let riskChartInst    = null;
let merchantChartInst= null;
let modelCompareInst = null;
let prRecallInst     = null;
let rfFiInst         = null;
let xgbFiInst        = null;
let merchantFraud    = { Grocery:0, 'E-Commerce':0, Restaurant:0, 'Fuel Station':0, Electronics:0, 'Travel/Hotel':0 };
let riskBuckets      = { LOW:0, MEDIUM:0, HIGH:0, CRITICAL:0 };
let trainingPoll     = null;

// ─── Chart defaults ────────────────────────────────────────
Chart.defaults.color           = '#94a3b8';
Chart.defaults.font.family     = 'Inter';
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const GRAD_PURPLE = (ctx) => {
  const g = ctx.createLinearGradient(0,0,0,300);
  g.addColorStop(0,'rgba(139,92,246,0.8)');
  g.addColorStop(1,'rgba(139,92,246,0.1)');
  return g;
};

const COLORS = {
  purple : 'rgba(139,92,246,1)',
  cyan   : 'rgba(34,211,238,1)',
  green  : 'rgba(74,222,128,1)',
  red    : 'rgba(248,113,113,1)',
  amber  : 'rgba(251,191,36,1)',
};

// ═══════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════
const SECTION_META = {
  dashboard : { title:'Live Dashboard',          sub:'Real-time financial transaction monitoring' },
  analyze   : { title:'Analyze Transaction',     sub:'Submit a transaction for instant AI fraud analysis' },
  models    : { title:'Model Performance',       sub:'Metrics and comparisons across all AI detectors' },
  history   : { title:'Transaction History',     sub:'Full log of analyzed transactions' },
  features  : { title:'Feature Importance',      sub:'Which signals matter most for fraud detection' },
};

function showSection(id, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`section-${id}`).classList.add('active');
  el.classList.add('active');
  const meta = SECTION_META[id];
  document.getElementById('pageTitle').textContent    = meta.title;
  document.getElementById('pageSubtitle').textContent = meta.sub;

  if (id === 'history')  fetchHistory();
  if (id === 'features') fetchFeatureImportance();
  if (id === 'models')   fetchMetrics();
}

// ═══════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkHealth() {
  const dot  = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');
  try {
    const r    = await fetch(`${API}/health`);
    const data = await r.json();
    dot.className  = 'status-dot online';
    text.textContent = `${data.models_loaded.length} models online`;
    dashStats.models = data.models_loaded.length;
    document.getElementById('kpiModels').textContent = data.models_loaded.length;
  } catch {
    dot.className    = 'status-dot offline';
    text.textContent = 'API offline';
  }
}

// ═══════════════════════════════════════════════════════════
// INITIAL CHARTS
// ═══════════════════════════════════════════════════════════
function initRiskChart() {
  const ctx = document.getElementById('riskChart').getContext('2d');
  riskChartInst = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low', 'Medium', 'High', 'Critical'],
      datasets:[{
        data: [1,0,0,0],
        backgroundColor: ['rgba(74,222,128,0.8)','rgba(251,191,36,0.8)','rgba(248,113,113,0.8)','rgba(239,68,68,0.9)'],
        borderColor: ['rgba(74,222,128,0.3)','rgba(251,191,36,0.3)','rgba(248,113,113,0.3)','rgba(239,68,68,0.3)'],
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      cutout: '65%',
      plugins: { legend: { position:'bottom' }, tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} transactions` } } },
    }
  });
}

function initMerchantChart() {
  const ctx = document.getElementById('merchantChart').getContext('2d');
  merchantChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(merchantFraud),
      datasets:[{
        label: 'Fraud Detected',
        data: Object.values(merchantFraud),
        backgroundColor: 'rgba(248,113,113,0.7)',
        borderColor: 'rgba(248,113,113,1)',
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive:true,
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ grid:{ color:'rgba(255,255,255,0.05)' } },
        y:{ grid:{ color:'rgba(255,255,255,0.05)' }, ticks:{ precision:0 } },
      }
    }
  });
}

function updateRiskChart() {
  if (!riskChartInst) return;
  riskChartInst.data.datasets[0].data = [riskBuckets.LOW, riskBuckets.MEDIUM, riskBuckets.HIGH, riskBuckets.CRITICAL];
  riskChartInst.update('none');
}

function updateMerchantChart() {
  if (!merchantChartInst) return;
  merchantChartInst.data.datasets[0].data = Object.values(merchantFraud);
  merchantChartInst.update('none');
}

// ═══════════════════════════════════════════════════════════
// SIMULATE TRANSACTIONS (Dashboard)
// ═══════════════════════════════════════════════════════════
async function simulateTransactions() {
  const btn = document.getElementById('simulateBtn');
  btn.disabled = true;
  btn.textContent = 'Running…';
  try {
    const r    = await fetch(`${API}/simulate?n=25`);
    const data = await r.json();

    dashStats.total += data.total;
    dashStats.fraud += data.fraud_detected;
    dashStats.legit += data.legit_count;

    document.getElementById('kpiTotal').textContent = dashStats.total.toLocaleString();
    document.getElementById('kpiFraud').textContent = dashStats.fraud.toLocaleString();
    document.getElementById('kpiLegit').textContent = dashStats.legit.toLocaleString();
    document.getElementById('kpiFraudRate').textContent = `${((dashStats.fraud/dashStats.total)*100).toFixed(1)}% fraud rate`;
    document.getElementById('kpiLegitRate').textContent = `${((dashStats.legit/dashStats.total)*100).toFixed(1)}% legitimate`;

    // Feed + charts
    const feed = document.getElementById('liveFeed');
    const emptyEl = feed.querySelector('.feed-empty');
    if (emptyEl) emptyEl.remove();

    data.transactions.forEach(txn => {
      const rl = txn.risk_level || 'LOW';
      riskBuckets[rl] = (riskBuckets[rl] || 0) + 1;
      if (txn.is_fraud && txn.merchant) {
        merchantFraud[txn.merchant] = (merchantFraud[txn.merchant] || 0) + 1;
      }
      const el = document.createElement('div');
      el.className = 'feed-item';
      const badgeCls = rl === 'LOW' ? 'legit' : rl === 'MEDIUM' ? 'medium' : 'fraud';
      el.innerHTML = `
        <span class="feed-id">${txn.id || 'TXN—'}</span>
        <span class="feed-merchant">${txn.merchant || '—'}</span>
        <span class="feed-amount">$${(txn.amount||0).toFixed(2)}</span>
        <span class="feed-badge ${badgeCls}">${txn.is_fraud ? '⚠ FRAUD' : '✓ LEGIT'} ${(txn.ensemble_score*100).toFixed(0)}%</span>
      `;
      feed.prepend(el);
      // Limit feed length
      while (feed.children.length > 60) feed.removeChild(feed.lastChild);
    });

    updateRiskChart();
    updateMerchantChart();
    toast(`Simulated ${data.total} transactions — ${data.fraud_detected} fraud detected`, 'success');
  } catch (e) {
    toast('Simulation failed. Is the API running?', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polygon points="5,3 19,12 5,21"/></svg> Simulate`;
  }
}

// ═══════════════════════════════════════════════════════════
// ANALYZE TRANSACTION (Single)
// ═══════════════════════════════════════════════════════════
function fillRandom() {
  const merchants = [0,1,2,3,4,5];
  const isFraud   = Math.random() < 0.3;
  document.getElementById('f_amount').value   = isFraud ? (Math.random()*5000+100).toFixed(2) : (Math.random()*500+10).toFixed(2);
  document.getElementById('f_hour').value     = isFraud ? Math.floor(Math.random()*5) : Math.floor(Math.random()*14)+8;
  document.getElementById('f_dow').value      = Math.floor(Math.random()*7);
  document.getElementById('f_merchant').value = merchants[Math.floor(Math.random()*6)];
  document.getElementById('f_freq').value     = isFraud ? Math.floor(Math.random()*15)+5 : Math.floor(Math.random()*5);
  document.getElementById('f_avg').value      = (Math.random()*400+50).toFixed(2);
  document.getElementById('f_dist').value     = isFraud ? (Math.random()*500+50).toFixed(1) : (Math.random()*20).toFixed(1);
  document.getElementById('f_foreign').value  = isFraud ? (Math.random()<0.5?1:0) : 0;
  document.getElementById('f_1h').value       = isFraud ? Math.floor(Math.random()*8)+2 : Math.floor(Math.random()*2);
  document.getElementById('f_24h').value      = isFraud ? Math.floor(Math.random()*20)+10 : Math.floor(Math.random()*6)+1;
  document.getElementById('f_age').value      = isFraud ? Math.floor(Math.random()*180)+1 : Math.floor(Math.random()*3000)+30;
  document.getElementById('f_limit').value    = [1000,2500,5000,10000][Math.floor(Math.random()*4)];
  document.getElementById('f_card').value     = isFraud ? 0 : 1;
  document.getElementById('f_pin').value      = isFraud ? 0 : 1;
  document.getElementById('f_online').value   = isFraud ? 1 : 0;
}

async function analyzeTransaction(e) {
  e.preventDefault();
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.textContent = 'Analyzing…';

  const payload = {
    amount:                 parseFloat(document.getElementById('f_amount').value),
    hour_of_day:            parseInt(document.getElementById('f_hour').value),
    day_of_week:            parseInt(document.getElementById('f_dow').value),
    merchant_category:      parseInt(document.getElementById('f_merchant').value),
    transaction_frequency:  parseInt(document.getElementById('f_freq').value),
    avg_transaction_amount: parseFloat(document.getElementById('f_avg').value),
    distance_from_home_km:  parseFloat(document.getElementById('f_dist').value),
    is_foreign_transaction: parseInt(document.getElementById('f_foreign').value),
    num_transactions_1h:    parseInt(document.getElementById('f_1h').value),
    num_transactions_24h:   parseInt(document.getElementById('f_24h').value),
    account_age_days:       parseInt(document.getElementById('f_age').value),
    credit_limit:           parseFloat(document.getElementById('f_limit').value),
    card_present:           parseInt(document.getElementById('f_card').value),
    pin_used:               parseInt(document.getElementById('f_pin').value),
    online_transaction:     parseInt(document.getElementById('f_online').value),
  };

  try {
    const r    = await fetch(`${API}/predict`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    if (!r.ok) { const err = await r.json(); throw new Error(err.detail || 'Server error'); }
    const data = await r.json();
    renderAnalysisResult(data);
  } catch (err) {
    toast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> Analyze Transaction`;
  }
}

function renderAnalysisResult(data) {
  document.getElementById('emptyResult').style.display   = 'none';
  document.getElementById('resultPanel').style.display   = 'block';

  const score    = data.ensemble_score || 0;
  const pct      = Math.round(score * 100);
  const riskLevel= data.risk_level || 'LOW';

  // Gauge
  const totalArcLen = Math.PI;
  const angle = (score * 180) - 180;  // -180 to 0 degrees
  const rad  = angle * Math.PI / 180;
  const cx = 100, cy = 100, r = 80;
  const ex = cx + r * Math.cos(rad);
  const ey = cy + r * Math.sin(rad);
  const largeArc = score > 0.5 ? 1 : 0;

  document.getElementById('gaugeArc').setAttribute('d',
    `M 20 100 A 80 80 0 ${largeArc} 1 ${ex.toFixed(1)} ${ey.toFixed(1)}`
  );
  document.getElementById('gaugeLabel').textContent = `${pct}%`;
  document.getElementById('riskText').textContent   = `${riskLevel} RISK`;

  // Badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = `${riskLevel} — ${data.is_fraud ? '⚠ FRAUD DETECTED' : '✓ LEGITIMATE'}`;
  badge.className = `risk-badge ${riskLevel.toLowerCase()}`;

  // Model scores
  const container = document.getElementById('modelScores');
  container.innerHTML = '';
  const modelColors = {
    isolation_forest: COLORS.cyan,
    random_forest:    COLORS.green,
    xgboost:          COLORS.purple,
    autoencoder:      COLORS.amber,
  };
  const modelLabels = {
    isolation_forest: 'Isolation Forest',
    random_forest:    'Random Forest',
    xgboost:          'XGBoost',
    autoencoder:      'Autoencoder',
  };

  Object.entries(data.predictions || {}).forEach(([name, pred]) => {
    if (pred.error) return;
    const prob  = pred.probability;
    const color = modelColors[name] || COLORS.purple;
    const label = modelLabels[name] || name;
    const barColor = prob >= 0.5 ? COLORS.red : (prob >= 0.25 ? COLORS.amber : COLORS.green);
    container.innerHTML += `
      <div class="model-score-row">
        <span class="model-score-name">${label}</span>
        <div class="model-score-bar-wrap">
          <div class="model-score-bar" style="width:${(prob*100).toFixed(1)}%;background:${barColor}"></div>
        </div>
        <span class="model-score-pct" style="color:${barColor}">${(prob*100).toFixed(1)}%</span>
      </div>
    `;
  });

  toast(data.is_fraud ? `⚠ Fraud detected! Risk: ${pct}%` : `✓ Transaction looks legitimate (${pct}% risk)`,
        data.is_fraud ? 'error' : 'success');
}

// ═══════════════════════════════════════════════════════════
// TRAIN MODELS
// ═══════════════════════════════════════════════════════════
async function trainModels() {
  const overlay = document.getElementById('trainOverlay');
  const bar     = document.getElementById('trainProgressBar');
  overlay.classList.add('active');
  bar.style.width = '0%';

  try {
    await fetch(`${API}/train`, { method: 'POST' });
    toast('Training started — this takes 2–5 minutes', 'info');

    // Animate progress bar
    let pct = 0;
    trainingPoll = setInterval(async () => {
      pct = Math.min(pct + Math.random() * 8, 95);
      bar.style.width = `${pct}%`;
      // Check if metrics are available (training done)
      try {
        const r = await fetch(`${API}/metrics`);
        if (r.ok) {
          clearInterval(trainingPoll);
          bar.style.width = '100%';
          setTimeout(() => {
            overlay.classList.remove('active');
            toast('✅ All models trained successfully!', 'success');
            checkHealth();
            fetchMetrics();
          }, 800);
        }
      } catch {}
    }, 5000);
  } catch {
    clearInterval(trainingPoll);
    overlay.classList.remove('active');
    toast('Training request failed. Is the API running?', 'error');
  }
}

// ═══════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════
async function fetchMetrics() {
  try {
    const r    = await fetch(`${API}/metrics`);
    if (!r.ok) { toast('No metrics yet. Train models first.', 'info'); return; }
    const data = await r.json();
    renderMetrics(data);
  } catch {
    toast('Could not load metrics', 'error');
  }
}

function renderMetrics(data) {
  const grid = document.getElementById('modelsGrid');
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = 'repeat(2, 1fr)';

  const ICONS = {
    isolation_forest: '🌲',
    random_forest:    '🌳',
    xgboost:          '⚡',
    autoencoder:      '🧠',
  };
  const PALETTE = {
    isolation_forest: COLORS.cyan,
    random_forest:    COLORS.green,
    xgboost:          COLORS.purple,
    autoencoder:      COLORS.amber,
  };
  const NAMES = {
    isolation_forest: 'Isolation Forest',
    random_forest:    'Random Forest',
    xgboost:          'XGBoost',
    autoencoder:      'Autoencoder (DL)',
  };

  Object.entries(data).forEach(([key, d]) => {
    const cm = d.confusion_matrix || [[0,0],[0,0]];
    const color = PALETTE[key] || COLORS.purple;
    const card = document.createElement('div');
    card.className = 'card glass model-metric-card';
    card.innerHTML = `
      <div class="model-metric-header">
        <h3>${ICONS[key] || '🤖'} ${NAMES[key] || key}</h3>
        <span class="train-time">${d.train_time || '—'}s train</span>
      </div>
      <div class="metrics-list">
        <div class="metric-item">
          <span class="m-label">Precision</span>
          <span class="m-value" style="color:${color}">${((d.precision||0)*100).toFixed(1)}%</span>
        </div>
        <div class="metric-item">
          <span class="m-label">Recall</span>
          <span class="m-value" style="color:${color}">${((d.recall||0)*100).toFixed(1)}%</span>
        </div>
        <div class="metric-item">
          <span class="m-label">F1-Score</span>
          <span class="m-value" style="color:${color}">${((d.f1_score||0)*100).toFixed(1)}%</span>
        </div>
        <div class="metric-item">
          <span class="m-label">ROC-AUC</span>
          <span class="m-value" style="color:${color}">${d.roc_auc ? (d.roc_auc*100).toFixed(1)+'%' : '—'}</span>
        </div>
      </div>
      <div class="confusion-matrix">
        <h4>Confusion Matrix</h4>
        <div class="cm-grid">
          <div class="cm-cell tn">${(cm[0]&&cm[0][0])||0}<div class="cm-label">TN</div></div>
          <div class="cm-cell fp">${(cm[0]&&cm[0][1])||0}<div class="cm-label">FP</div></div>
          <div class="cm-cell fn">${(cm[1]&&cm[1][0])||0}<div class="cm-label">FN</div></div>
          <div class="cm-cell tp">${(cm[1]&&cm[1][1])||0}<div class="cm-label">TP</div></div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });

  // Comparison chart
  document.getElementById('metricsCharts').style.display = 'grid';
  const names  = Object.keys(data).map(k => NAMES[k] || k);
  const f1s    = Object.values(data).map(d => +((d.f1_score||0)*100).toFixed(2));
  const aucs   = Object.values(data).map(d => +((d.roc_auc||0)*100).toFixed(2));
  const precs  = Object.values(data).map(d => +((d.precision||0)*100).toFixed(2));
  const recs   = Object.values(data).map(d => +((d.recall||0)*100).toFixed(2));

  if (modelCompareInst) modelCompareInst.destroy();
  modelCompareInst = new Chart(document.getElementById('modelCompareChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        { label:'F1-Score (%)',  data: f1s,  backgroundColor:'rgba(139,92,246,0.7)', borderColor:COLORS.purple, borderWidth:2, borderRadius:6 },
        { label:'ROC-AUC (%)',   data: aucs, backgroundColor:'rgba(34,211,238,0.7)',  borderColor:COLORS.cyan,   borderWidth:2, borderRadius:6 },
      ],
    },
    options: { responsive:true, scales:{ y:{ beginAtZero:true, max:100, grid:{color:'rgba(255,255,255,0.05)'} }, x:{grid:{color:'rgba(255,255,255,0.05)'}} } }
  });

  if (prRecallInst) prRecallInst.destroy();
  prRecallInst = new Chart(document.getElementById('prRecallChart').getContext('2d'), {
    type: 'radar',
    data: {
      labels: names,
      datasets: [
        { label:'Precision (%)', data: precs, borderColor:COLORS.green, backgroundColor:'rgba(74,222,128,0.1)', pointBackgroundColor:COLORS.green },
        { label:'Recall (%)',    data: recs,  borderColor:COLORS.red,   backgroundColor:'rgba(248,113,113,0.1)', pointBackgroundColor:COLORS.red },
      ]
    },
    options: {
      responsive:true,
      scales:{ r:{ beginAtZero:true, max:100, grid:{color:'rgba(255,255,255,0.08)'}, angleLines:{color:'rgba(255,255,255,0.08)'} } }
    }
  });
}

// ═══════════════════════════════════════════════════════════
// TRANSACTION HISTORY
// ═══════════════════════════════════════════════════════════
async function fetchHistory() {
  try {
    const r    = await fetch(`${API}/transactions?limit=100`);
    const data = await r.json();
    const tbody= document.getElementById('historyBody');
    if (!data.transactions || data.transactions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No transactions recorded yet.</td></tr>';
      return;
    }
    tbody.innerHTML = data.transactions.slice().reverse().map(t => `
      <tr>
        <td><code>${t.id || '—'}</code></td>
        <td>${t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '—'}</td>
        <td>$${(t.amount||0).toFixed(2)}</td>
        <td>${t.merchant || '—'}</td>
        <td><span style="font-family:'JetBrains Mono',monospace;color:${riskColor(t.risk_level)}">${((t.ensemble_score||0)*100).toFixed(1)}%</span></td>
        <td><span class="risk-pill ${t.risk_level||'LOW'}">${t.risk_level||'LOW'}</span></td>
        <td class="${t.is_fraud?'verdict-fraud':'verdict-legit'}">${t.is_fraud?'⚠ Fraud':'✓ Legit'}</td>
      </tr>
    `).join('');
  } catch {
    toast('Could not load history', 'error');
  }
}

function riskColor(lvl) {
  return {LOW:'#4ade80',MEDIUM:'#fbbf24',HIGH:'#f87171',CRITICAL:'#fca5a5'}[lvl] || '#94a3b8';
}

// ═══════════════════════════════════════════════════════════
// FEATURE IMPORTANCE
// ═══════════════════════════════════════════════════════════
const FEATURE_DESCRIPTIONS = {
  amount:                  'Raw transaction amount in USD',
  hour_of_day:             'Hour the transaction occurred (0-23)',
  day_of_week:             'Day of the week (0=Monday, 6=Sunday)',
  merchant_category:       'Type of merchant (grocery, e-commerce, etc.)',
  transaction_frequency:   'Number of transactions in the past 7 days',
  avg_transaction_amount:  'User\'s historical average spend',
  distance_from_home_km:   'Physical distance of txn from home address',
  is_foreign_transaction:  'Whether the transaction is in a foreign country',
  num_transactions_1h:     'Number of transactions in the last 1 hour',
  num_transactions_24h:    'Number of transactions in the last 24 hours',
  account_age_days:        'Age of the account in days',
  credit_limit:            'User\'s credit limit on the card',
  card_present:            'Whether the physical card was used',
  pin_used:                'Whether a PIN was entered',
  online_transaction:      'Whether transaction was online',
  amount_to_avg_ratio:     'Ratio of txn amount to user\'s average',
  credit_utilization:      'Fraction of credit limit being used',
  is_night_transaction:    'Transaction occurred between 10 PM and 5 AM',
  high_velocity:           'More than 3 transactions in the last hour',
  is_weekend:              'Transaction occurred on Saturday or Sunday',
};

async function fetchFeatureImportance() {
  // Feature descriptions
  const descGrid = document.getElementById('featureDescGrid');
  descGrid.innerHTML = Object.entries(FEATURE_DESCRIPTIONS).map(([k,v]) => `
    <div class="feature-desc-item">
      <div class="fd-name">${k}</div>
      <div class="fd-desc">${v}</div>
    </div>
  `).join('');

  try {
    const r    = await fetch(`${API}/feature_importance`);
    if (!r.ok) return;
    const data = await r.json();

    if (data.random_forest) renderFiChart('rfFiChart', data.random_forest, rfFiInst, 'rgba(74,222,128,0.75)', 'rgba(74,222,128,1)');
    if (data.xgboost)       renderFiChart('xgbFiChart', data.xgboost, xgbFiInst, 'rgba(139,92,246,0.75)', 'rgba(139,92,246,1)');
  } catch {}
}

function renderFiChart(canvasId, fiData, existingChart, bgColor, borderColor) {
  if (existingChart) existingChart.destroy();
  const top10  = fiData.slice(0, 10);
  const labels = top10.map(d => d.feature);
  const values = top10.map(d => +(d.importance * 100).toFixed(3));

  const chart = new Chart(document.getElementById(canvasId).getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label:'Importance (%)', data:values, backgroundColor:bgColor, borderColor, borderWidth:2, borderRadius:6 }],
    },
    options: {
      indexAxis:'y',
      responsive:true,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ grid:{color:'rgba(255,255,255,0.05)'} },
        y:{ grid:{color:'rgba(255,255,255,0.05)'}, ticks:{font:{size:11}} },
      }
    }
  });
  if (canvasId === 'rfFiChart')  rfFiInst  = chart;
  if (canvasId === 'xgbFiChart') xgbFiInst = chart;
}

// ═══════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════
function clearFeed() {
  document.getElementById('liveFeed').innerHTML = '<div class="feed-empty">Feed cleared</div>';
}

function toast(msg, type = 'info') {
  const icons = {
    success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="18" height="18"><polyline points="20,6 9,17 4,12"/></svg>',
    error:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="18" height="18"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    info:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="18" height="18"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
  };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span class="toast-icon ${type}">${icons[type]||''}</span><span>${msg}</span>`;
  document.getElementById('toastContainer').appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  initRiskChart();
  initMerchantChart();
  setInterval(checkHealth, 30000);  // Refresh health every 30s
  toast('FraudShield AI ready. Train models or run a simulation!', 'info');
});
