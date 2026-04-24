const state = { data: null, selectedStation: null };
const runtime = {
  online_app_url: null,
  api_doc_url: null,
  service_status: '待配置',
};

function fmt(n, d = 3) { return Number(n).toFixed(d); }
function riskLabel(i) { return state.data.risk_labels[i] || '未知'; }
function riskColor(i) { return ['#22c55e', '#f59e0b', '#f97316', '#ef4444'][i] || '#60a5fa'; }

async function loadData() {
  const r = await fetch('./assets/demo_data.json');
  return r.json();
}

async function loadSiteConfig() {
  try {
    const r = await fetch('./assets/site_config.json');
    if (!r.ok) return;
    const cfg = await r.json();
    runtime.online_app_url = cfg.online_app_url || null;
    runtime.api_doc_url = cfg.api_doc_url || null;
    runtime.service_status = cfg.service_status || '待配置';
  } catch (_) {}
}

function hydrateHero() {
  const reg = state.data.metrics.dustriskformer.regression;
  const evt = state.data.event_summary.event_level_macro;
  document.getElementById('heroMae').textContent = fmt(reg.mae);
  document.getElementById('heroRmse').textContent = fmt(reg.rmse);
  document.getElementById('heroR2').textContent = fmt(reg.r2);
  document.getElementById('heroHit').textContent = fmt(evt.event_hit_rate);
  document.getElementById('updatedAt').textContent = new Date().toLocaleString('zh-CN');
  document.getElementById('serviceStatus').textContent = runtime.service_status;
}

function renderModelCompare() {
  const m = state.data.metrics;
  const names = ['dustriskformer', 'cnn_lstm', 'lstm'];
  const labels = ['主模型', 'CNN-LSTM', 'LSTM'];
  const maes = names.map((k) => m[k].regression.mae);
  const f1s = names.map((k) => m[k].warning_binary.f1);

  const c = echarts.init(document.getElementById('modelCompare'));
  c.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d7e7ff' } },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#b9cdeb' } },
    yAxis: [
      { type: 'value', name: 'MAE', axisLabel: { color: '#b9cdeb' } },
      { type: 'value', name: 'F1', min: 0, max: 1, axisLabel: { color: '#b9cdeb' } },
    ],
    series: [
      { name: 'MAE', type: 'bar', data: maes, itemStyle: { color: '#60a5fa' } },
      { name: 'Warning F1', type: 'line', yAxisIndex: 1, data: f1s, smooth: true, lineStyle: { color: '#4ade80' } },
    ],
  });
}

function renderEventSummary() {
  const e = state.data.event_summary.event_level_macro;
  const wrap = document.getElementById('eventSummary');
  const items = [
    ['事件命中率', e.event_hit_rate],
    ['起报提前量误差', e.onset_lead_error],
    ['持续时长误差', e.duration_error],
    ['阈值跨越F1', e.crossing_f1],
  ];
  wrap.innerHTML = items
    .map(
      ([k, v]) => `
    <div class="event-item">
      <div class="name">${k}</div>
      <div class="value">${fmt(v)}</div>
    </div>
  `
    )
    .join('');
}

function initStationSelect() {
  const sel = document.getElementById('stationSelect');
  sel.innerHTML = '';
  state.data.stations.forEach((s) => {
    const op = document.createElement('option');
    op.value = s.station_id;
    op.textContent = `${s.station_name} (${s.station_id})`;
    sel.appendChild(op);
  });
  state.selectedStation = state.data.stations[0].station_id;
  sel.value = state.selectedStation;
  sel.addEventListener('change', () => {
    state.selectedStation = sel.value;
    runPredictionDemo();
  });
}

function simulateForecast(stationId, windInput, rhInput) {
  const base = state.data.base_profiles[stationId];
  const g = state.data.sensitivity.wind_gain;
  const p = state.data.sensitivity.rh_penalty;

  return base.horizons.map((h, i) => {
    const bw = base.wind[i];
    const bp = base.warn_prob[i];
    const wind = bw + g[i] * (windInput - 8) - 0.028 * (rhInput - 35);
    const prob = Math.max(0.01, Math.min(0.99, bp + 0.036 * (windInput - 8) - p[i] * (rhInput - 35)));
    const risk = prob > 0.8 ? 3 : prob > 0.6 ? 2 : prob > 0.35 ? 1 : 0;
    return {
      h,
      wind: +wind.toFixed(3),
      wind_p10: +(wind * 0.88).toFixed(3),
      wind_p90: +(wind * 1.12).toFixed(3),
      prob: +prob.toFixed(3),
      prob_p10: +Math.max(0, prob - 0.08).toFixed(3),
      prob_p90: +Math.min(1, prob + 0.08).toFixed(3),
      risk,
    };
  });
}

function renderForecast(forecast) {
  const c = echarts.init(document.getElementById('forecastChart'));
  c.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d7e7ff' } },
    xAxis: { type: 'category', data: forecast.map((x) => `${x.h}h`), axisLabel: { color: '#b9cdeb' } },
    yAxis: [
      { type: 'value', name: '风速', axisLabel: { color: '#b9cdeb' } },
      { type: 'value', name: '预警概率', min: 0, max: 1, axisLabel: { color: '#b9cdeb' } },
    ],
    series: [
      { name: '风速', type: 'line', smooth: true, data: forecast.map((x) => x.wind), lineStyle: { color: '#60a5fa' } },
      { name: '预警概率', type: 'bar', yAxisIndex: 1, data: forecast.map((x) => x.prob), itemStyle: { color: '#f59e0b' } },
    ],
  });

  document.getElementById('forecastTable').innerHTML = `
    <thead><tr><th>时效</th><th>风速</th><th>风速区间</th><th>风险等级</th><th>预警概率</th><th>概率区间</th></tr></thead>
    <tbody>
      ${forecast
        .map(
          (x) => `
        <tr>
          <td>${x.h}h</td>
          <td>${x.wind}</td>
          <td>${x.wind_p10} ~ ${x.wind_p90}</td>
          <td style="color:${riskColor(x.risk)}">${riskLabel(x.risk)}</td>
          <td>${x.prob}</td>
          <td>${x.prob_p10} ~ ${x.prob_p90}</td>
        </tr>
      `
        )
        .join('')}
    </tbody>
  `;
}

function renderMap() {
  const chart = echarts.init(document.getElementById('mapChart'));
  const data = state.data.stations.map((s) => ({
    name: s.station_name,
    value: [s.lon, s.lat, s.warn_prob_3h, s.risk_3h],
  }));

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2000);

  fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json', { signal: controller.signal })
    .then((r) => r.json())
    .then((geo) => {
      clearTimeout(timeoutId);
      echarts.registerMap('china', geo);
      chart.setOption({
        backgroundColor: 'transparent',
        geo: {
          map: 'china',
          roam: true,
          itemStyle: { areaColor: '#1a2c49', borderColor: '#47608a' },
          emphasis: { itemStyle: { areaColor: '#2a4472' } },
        },
        tooltip: {
          formatter: (p) => `${p.name}<br/>风险: ${riskLabel((p.value || [])[3] || 0)}<br/>3h预警概率: ${fmt((p.value || [])[2] || 0)}`,
        },
        visualMap: {
          min: 0,
          max: 1,
          orient: 'horizontal',
          left: 'center',
          bottom: 10,
          textStyle: { color: '#cfe1ff' },
          calculable: false,
        },
        series: [
          {
            type: 'scatter',
            coordinateSystem: 'geo',
            data,
            symbolSize: (v) => 10 + v[2] * 18,
            itemStyle: { color: (p) => riskColor(p.data.value[3]), borderColor: '#fff', borderWidth: 1 },
          },
        ],
      });
    })
    .catch(() => {
      clearTimeout(timeoutId);
      chart.setOption({
        xAxis: { type: 'value', name: '经度', axisLabel: { color: '#b9cdeb' } },
        yAxis: { type: 'value', name: '纬度', axisLabel: { color: '#b9cdeb' } },
        tooltip: { formatter: (p) => `${p.name}<br/>风险: ${riskLabel((p.value || [])[3] || 0)}` },
        series: [
          {
            type: 'scatter',
            data: data.map((d) => ({ name: d.name, value: d.value })),
            symbolSize: (v) => 10 + v[2] * 18,
            itemStyle: { color: (p) => riskColor(p.data.value[3]) },
          },
        ],
      });
    });
}

function renderReplay(stationId) {
  const rep = state.data.replay[stationId]['3'];

  const cw = echarts.init(document.getElementById('replayWindChart'));
  cw.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d7e7ff' } },
    xAxis: { type: 'category', data: rep.sample_idx, axisLabel: { show: false, color: '#b9cdeb' } },
    yAxis: { type: 'value', axisLabel: { color: '#b9cdeb' } },
    series: [
      { name: '真值', type: 'line', data: rep.y_wind_true, lineStyle: { color: '#4ade80' } },
      { name: '预测', type: 'line', data: rep.y_wind_pred, lineStyle: { color: '#60a5fa' } },
    ],
  });

  const cp = echarts.init(document.getElementById('replayWarnChart'));
  cp.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d7e7ff' } },
    xAxis: { type: 'category', data: rep.sample_idx, axisLabel: { show: false, color: '#b9cdeb' } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#b9cdeb' } },
    series: [
      { name: '预警真值', type: 'line', step: 'middle', data: rep.y_warn_true, lineStyle: { color: '#f97316' } },
      { name: '预警概率', type: 'line', data: rep.y_warn_prob, lineStyle: { color: '#facc15' } },
    ],
  });
}

function runPredictionDemo() {
  const sid = state.selectedStation;
  const wind = Number(document.getElementById('windInput').value || 8);
  const rh = Number(document.getElementById('rhInput').value || 35);
  const f = simulateForecast(sid, wind, rh);
  renderForecast(f);
  renderReplay(sid);
}

function bindButtons() {
  document.getElementById('runBtn').addEventListener('click', runPredictionDemo);

  const apiBtn = document.getElementById('apiDocBtn');
  const appBtn = document.getElementById('onlineAppBtn');
  const ctaDemo = document.getElementById('ctaDemo');
  const ctaApi = document.getElementById('ctaApi');

  if (runtime.api_doc_url) {
    apiBtn.href = runtime.api_doc_url;
    ctaApi.href = runtime.api_doc_url;
  }
  if (runtime.online_app_url) {
    appBtn.href = runtime.online_app_url;
    ctaDemo.href = runtime.online_app_url;
    ctaDemo.target = '_blank';
    ctaDemo.rel = 'noopener';
  }
}

function onResize() {
  ['modelCompare', 'forecastChart', 'mapChart', 'replayWindChart', 'replayWarnChart'].forEach((id) => {
    const inst = echarts.getInstanceByDom(document.getElementById(id));
    if (inst) inst.resize();
  });
}

(async function bootstrap() {
  await loadSiteConfig();
  state.data = await loadData();
  hydrateHero();
  renderModelCompare();
  renderEventSummary();
  initStationSelect();
  renderMap();
  bindButtons();
  runPredictionDemo();
  window.addEventListener('resize', onResize);
})();
