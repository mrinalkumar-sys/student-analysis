/**
 * Student Performance Analytics Dashboard Frontend Client
 */

let currentPage = 1;
const pageSize = 15;
let charts = {};

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.nav-tab').forEach(el => {
    el.classList.remove('active');
    el.classList.add('text-slate-400');
  });

  const activeContent = document.getElementById(`tab-content-${tabName}`);
  const activeBtn = document.getElementById(`tab-btn-${tabName}`);
  if (activeContent) activeContent.classList.remove('hidden');
  if (activeBtn) {
    activeBtn.classList.add('active');
    activeBtn.classList.remove('text-slate-400');
  }

  if (tabName === 'eda' && Object.keys(charts).length === 0) {
    loadEdaCharts();
  } else if (tabName === 'models') {
    loadModelsInfo();
  }
}

// 1. Load Overview Metrics & Descriptive Statistics
async function loadOverview() {
  try {
    const res = await fetch('/api/data/summary');
    const data = await res.json();

    document.getElementById('metric-total-records').innerText = data.total_records.toLocaleString();
    document.getElementById('metric-math-mean').innerText = (data.mean.MathScore || 0).toFixed(1);
    document.getElementById('metric-math-std').innerText = (data.std_dev.MathScore || 0).toFixed(1);

    document.getElementById('metric-reading-mean').innerText = (data.mean.ReadingScore || 0).toFixed(1);
    document.getElementById('metric-reading-std').innerText = (data.std_dev.ReadingScore || 0).toFixed(1);

    document.getElementById('metric-writing-mean').innerText = (data.mean.WritingScore || 0).toFixed(1);
    document.getElementById('metric-writing-std').innerText = (data.std_dev.WritingScore || 0).toFixed(1);

    // Populate Descriptive Stats Table (Cell 9)
    const tbody = document.getElementById('descriptive-stats-body');
    const rows = [
      { label: 'Mean', data: data.mean },
      { label: 'Median', data: data.median },
      { label: 'Mode', data: data.mode },
      { label: 'Standard Deviation', data: data.std_dev }
    ];

    tbody.innerHTML = rows.map(r => `
      <tr class="hover:bg-slate-800/40 transition">
        <td class="px-4 py-3 font-semibold text-white">${r.label}</td>
        <td class="px-4 py-3 text-cyan-400">${r.data.MathScore !== undefined ? r.data.MathScore.toFixed(2) : '--'}</td>
        <td class="px-4 py-3 text-emerald-400">${r.data.ReadingScore !== undefined ? r.data.ReadingScore.toFixed(2) : '--'}</td>
        <td class="px-4 py-3 text-amber-400">${r.data.WritingScore !== undefined ? r.data.WritingScore.toFixed(2) : '--'}</td>
        <td class="px-4 py-3 text-slate-300">${r.data.NrSiblings !== undefined ? r.data.NrSiblings.toFixed(2) : '--'}</td>
      </tr>
    `).join('');

  } catch (err) {
    console.error('Error loading summary stats:', err);
  }
}

// 2. Fetch Paginated Records
async function fetchRecords(page = 1) {
  currentPage = page;
  const search = document.getElementById('record-search-input').value.trim();
  const tbody = document.getElementById('records-table-body');
  tbody.innerHTML = '<tr><td colspan="11" class="px-4 py-6 text-center text-slate-500">Loading records...</td></tr>';

  try {
    const url = `/api/data/records?page=${page}&page_size=${pageSize}` + (search ? `&search=${encodeURIComponent(search)}` : '');
    const res = await fetch(url);
    const data = await res.json();

    document.getElementById('pagination-info').innerText = `Showing page ${data.page} of ${data.total_pages} (${data.total.toLocaleString()} records)`;
    document.getElementById('prev-page-btn').disabled = data.page <= 1;
    document.getElementById('next-page-btn').disabled = data.page >= data.total_pages;

    if (data.records.length === 0) {
      tbody.innerHTML = '<tr><td colspan="11" class="px-4 py-6 text-center text-slate-500">No matching student records found.</td></tr>';
      return;
    }

    tbody.innerHTML = data.records.map(r => `
      <tr class="hover:bg-slate-800/50 transition">
        <td class="px-3 py-2 text-slate-300">${r.Gender || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.EthnicGroup || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.ParentEduc || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.LunchType || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.TestPrep || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.ParentMaritalStatus || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.PracticeSport || '-'}</td>
        <td class="px-3 py-2 text-slate-300">${r.WklyStudyHours || '-'}</td>
        <td class="px-3 py-2 font-bold text-cyan-400">${r.MathScore}</td>
        <td class="px-3 py-2 font-bold text-emerald-400">${r.ReadingScore}</td>
        <td class="px-3 py-2 font-bold text-amber-400">${r.WritingScore}</td>
      </tr>
    `).join('');

  } catch (err) {
    console.error('Error fetching records:', err);
  }
}

function prevPage() {
  if (currentPage > 1) fetchRecords(currentPage - 1);
}

function nextPage() {
  fetchRecords(currentPage + 1);
}

// 3. Load EDA Charts
async function loadEdaCharts() {
  try {
    const res = await fetch('/api/analytics/demographics');
    const data = await res.json();

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } }
      },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#334155' } },
        y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#334155' } }
      }
    };

    // Helper to create 3-bar score comparisons
    function makeScoresChart(canvasId, demoKey, title) {
      const demo = data[demoKey];
      if (!demo) return;

      const ctx = document.getElementById(canvasId).getContext('2d');
      charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: demo.categories,
          datasets: [
            { label: 'Math', data: demo.math_mean, backgroundColor: '#06b6d4' },
            { label: 'Reading', data: demo.reading_mean, backgroundColor: '#10b981' },
            { label: 'Writing', data: demo.writing_mean, backgroundColor: '#f59e0b' }
          ]
        },
        options: chartOptions
      });
    }

    makeScoresChart('chart-gender', 'Gender', 'Gender Comparison');
    makeScoresChart('chart-parent-educ', 'ParentEduc', 'Parental Education');
    makeScoresChart('chart-lunch-type', 'LunchType', 'Lunch Type');
    makeScoresChart('chart-sports', 'PracticeSport', 'Sports Frequency');
    makeScoresChart('chart-ethnic', 'EthnicGroup', 'Ethnic Groups');

    // Correlation Table
    loadCorrelationTable();

  } catch (err) {
    console.error('Error loading EDA charts:', err);
  }
}

async function loadCorrelationTable() {
  try {
    const res = await fetch('/api/analytics/correlations');
    const corr = await res.json();
    const tbody = document.getElementById('correlation-table-body');
    const cols = corr.columns;

    tbody.innerHTML = cols.map(rowCol => `
      <tr class="hover:bg-slate-800/40">
        <td class="p-3 text-left font-semibold text-white">${rowCol}</td>
        ${cols.map(c => {
          const val = corr.matrix[rowCol][c];
          const colorClass = val >= 0.9 ? 'text-emerald-400 font-bold' : (val >= 0.8 ? 'text-indigo-400' : 'text-slate-300');
          return `<td class="p-3 ${colorClass}">${val.toFixed(4)}</td>`;
        }).join('')}
      </tr>
    `).join('');
  } catch (err) {
    console.error('Error loading correlations:', err);
  }
}

// 4. Hypothesis Testing
function updateHypoGroupOptions() {
  const factor = document.getElementById('hypo-factor').value;
  const g1 = document.getElementById('hypo-group1');
  const g2 = document.getElementById('hypo-group2');

  const optionsMap = {
    'LunchType': ['standard', 'free/reduced'],
    'TestPrep': ['completed', 'none'],
    'Gender': ['female', 'male'],
    'PracticeSport': ['regularly', 'never']
  };

  const opts = optionsMap[factor] || ['Option 1', 'Option 2'];
  g1.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
  g2.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
  if (opts.length > 1) g2.selectedIndex = 1;
}

async function executeHypothesisTest(e) {
  e.preventDefault();
  const factor = document.getElementById('hypo-factor').value;
  const g1 = document.getElementById('hypo-group1').value;
  const g2 = document.getElementById('hypo-group2').value;
  const score = document.getElementById('hypo-score').value;

  try {
    const res = await fetch('/api/analytics/hypothesis-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        group_column: factor,
        group1_val: g1,
        group2_val: g2,
        score_column: score,
        alpha: 0.05
      })
    });

    const data = await res.json();
    const resultCard = document.getElementById('hypo-result-card');
    resultCard.classList.remove('hidden');

    const badge = document.getElementById('hypo-decision-badge');
    if (data.reject_null) {
      badge.className = 'badge-tag bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs';
      badge.innerText = 'Reject Null Hypothesis (Significant)';
    } else {
      badge.className = 'badge-tag bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs';
      badge.innerText = 'Fail to Reject Null Hypothesis';
    }

    document.getElementById('hypo-t-stat').innerText = data.t_statistic;
    document.getElementById('hypo-p-val').innerText = data.p_value_formatted;
    document.getElementById('hypo-mean-diff').innerText = data.mean_difference > 0 ? `+${data.mean_difference}` : data.mean_difference;
    document.getElementById('hypo-cohens-d').innerText = data.cohens_d;
    document.getElementById('hypo-conclusion-text').innerText = data.conclusion;

  } catch (err) {
    console.error('Hypothesis test error:', err);
    alert('Hypothesis test failed. Check console.');
  }
}

// 5. ML Inferences
async function predictScore(e) {
  e.preventDefault();
  const payload = {
    Gender: document.getElementById('reg-gender').value,
    EthnicGroup: document.getElementById('reg-ethnic').value,
    ParentEduc: document.getElementById('reg-parent-educ').value,
    LunchType: document.getElementById('reg-lunch').value,
    ReadingScore: parseFloat(document.getElementById('reg-reading').value),
    WritingScore: parseFloat(document.getElementById('reg-writing').value)
  };

  try {
    const res = await fetch('/api/predict/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    const box = document.getElementById('reg-output-box');
    box.classList.remove('hidden');
    document.getElementById('reg-predicted-val').innerText = result.predicted_score;
  } catch (err) {
    console.error('Score prediction error:', err);
  }
}

async function predictGrade(e) {
  e.preventDefault();
  const payload = {
    PracticeSport: document.getElementById('clf-sport').value,
    TestPrep: document.getElementById('clf-testprep').value,
    WklyStudyHours: document.getElementById('clf-study').value,
    NrSiblings: parseFloat(document.getElementById('clf-siblings').value)
  };

  try {
    const res = await fetch('/api/predict/grade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    const box = document.getElementById('clf-output-box');
    box.classList.remove('hidden');

    document.getElementById('clf-grade-badge').innerText = result.predicted_grade;
    document.getElementById('clf-confidence').innerText = `${(result.confidence * 100).toFixed(1)}%`;
    document.getElementById('clf-grade-desc').innerText = `Estimated academic performance: Grade ${result.predicted_grade}`;

    const bars = document.getElementById('grade-prob-bars');
    bars.innerHTML = Object.entries(result.probabilities).map(([grd, prob]) => `
      <span class="px-1.5 py-0.5 bg-slate-800 rounded font-mono">${grd}: ${(prob * 100).toFixed(0)}%</span>
    `).join('');

  } catch (err) {
    console.error('Grade prediction error:', err);
  }
}

// 6. Models & Data Info
async function loadModelsInfo() {
  try {
    const res = await fetch('/api/models/info');
    const data = await res.json();

    const regMetrics = document.getElementById('reg-metrics-content');
    regMetrics.innerHTML = `
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Algorithm:</span>
        <span class="font-medium text-white">${data.regression.algorithm}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Mean Squared Error (MSE):</span>
        <span class="font-bold text-indigo-400">${data.regression.metrics.mean_squared_error}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">R² Determination Score:</span>
        <span class="font-bold text-emerald-400">${data.regression.metrics.r2_score}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Root MSE (RMSE):</span>
        <span class="font-medium text-white">${data.regression.metrics.root_mean_squared_error}</span>
      </div>
    `;

    const clfMetrics = document.getElementById('clf-metrics-content');
    clfMetrics.innerHTML = `
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Algorithm:</span>
        <span class="font-medium text-white">${data.classification.algorithm}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Classification Accuracy:</span>
        <span class="font-bold text-cyan-400">${(data.classification.metrics.accuracy * 100).toFixed(1)}%</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Macro Average F1:</span>
        <span class="font-bold text-emerald-400">${data.classification.metrics.macro_avg_f1}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Grade Categories:</span>
        <span class="font-medium text-white">${data.classification.metrics.classes.join(', ')}</span>
      </div>
    `;

  } catch (err) {
    console.error('Error loading model info:', err);
  }
}

// 7. Upload Dataset
async function uploadDataset(e) {
  e.preventDefault();
  const fileInput = document.getElementById('csv-file-input');
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  const status = document.getElementById('upload-status');
  status.innerHTML = '<span class="text-indigo-400"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Uploading and retraining models...</span>';

  try {
    const res = await fetch('/api/data/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    status.innerHTML = `<span class="text-emerald-400"><i class="fa-solid fa-circle-check mr-1"></i> ${result.message}</span>`;
    loadOverview();
    fetchRecords(1);
    loadModelsInfo();
  } catch (err) {
    status.innerHTML = '<span class="text-rose-400">Upload failed. Please check file format.</span>';
  }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
  fetchRecords(1);
});
