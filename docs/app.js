const state = {
  data: null,
  cases: [],
  selectedStation: null,
  selectedReplayDate: null,
  mapHorizon: 3,
  mapLayer: 'risk',
  replayTimer: null,
};

const runtime = { online_app_url: null, api_doc_url: null, service_status: '待配置' };
const RISK_LABELS = ['低风险', '中风险', '高风险', '极高风险'];

function fmt(n, d = 3) { return Number(n).toFixed(d); }
function riskColor(i) { return ['#22c55e', '#f59e0b', '#f97316', '#ef4444'][i] || '#60a5fa'; }
function riskLabel(i) { return RISK_LABELS[i] || '未知'; }

async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`failed: ${path}`);
  return r.json();
}

async function loadSiteConfig() {
  try {
    const cfg = await fetchJSON('./assets/site_config.json');
    runtime.online_app_url = cfg.online_app_url || null;
    runtime.api_doc_url = cfg.api_doc_url || null;
    runtime.service_status = cfg.service_status || '待配置';
  } catch (_) {}
}

function sliceByCase(rep, caseIdx) {
  const n = rep.sample_idx.length;
  const chunk = Math.max(48, Math.floor(n / Math.max(1, state.cases.length)));
  const start = Math.min(caseIdx * chunk, Math.max(0, n - chunk));
  const end = Math.min(n, start + chunk);
  const cut = (arr) => arr.slice(start, end);
  return {
    sample_idx: cut(rep.sample_idx),
    y_wind_true: cut(rep.y_wind_true),
    y_wind_pred: cut(rep.y_wind_pred),
    y_warn_true: cut(rep.y_warn_true),
    y_warn_prob: cut(rep.y_warn_prob),
  };
}

function stationProfile(stationId, h) {
  const p = state.data.base_profiles[stationId];
  const idx = p.horizons.indexOf(Number(h));
  return {
    wind: p.wind[idx],
    warn: p.warn_prob[idx],
    risk: p.risk[idx],
    wind_p10: p.wind[idx] * 0.88,
    wind_p90: p.wind[idx] * 1.12,
    warn_p10: Math.max(0, p.warn_prob[idx] - 0.08),
    warn_p90: Math.min(1, p.warn_prob[idx] + 0.08),
  };
}

function hydrateHero() {
  const reg = state.data.metrics.dustriskformer.regression;
  const warn = state.data.metrics.dustriskformer.warning_binary;
  const evt = state.data.event_summary.event_level_macro;

  document.getElementById('heroMae').textContent = fmt(reg.mae);
  document.getElementById('heroRmse').textContent = fmt(reg.rmse);
  document.getElementById('heroR2').textContent = fmt(reg.r2);
  document.getElementById('heroHit').textContent = fmt(evt.event_hit_rate);
  document.getElementById('kpiF1').textContent = fmt(warn.f1);
  document.getElementById('kpiAuc').textContent = fmt(warn.roc_auc);
  document.getElementById('kpiPrauc').textContent = fmt(warn.pr_auc);
  document.getElementById('serviceStatus').textContent = runtime.service_status;
  document.getElementById('updatedAt').textContent = new Date().toLocaleString('zh-CN');
}

function renderModelCompare() {
  const m = state.data.metrics;
  const names = ['dustriskformer', 'cnn_lstm', 'lstm'];
  const labels = ['主模型', 'CNN-LSTM', 'LSTM'];

  const c = echarts.init(document.getElementById('modelCompare'));
  c.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d7e7ff' } },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#bfd4f6' } },
    yAxis: [
      { type: 'value', name: 'MAE', axisLabel: { color: '#bfd4f6' } },
      { type: 'value', name: 'F1', min: 0, max: 1, axisLabel: { color: '#bfd4f6' } },
    ],
    series: [
      { name: 'MAE', type: 'bar', data: names.map((k) => m[k].regression.mae), itemStyle: { color: '#60a5fa' } },
      { name: 'Warning F1', type: 'line', yAxisIndex: 1, smooth: true, data: names.map((k) => m[k].warning_binary.f1), lineStyle: { color: '#34d399' } },
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
  wrap.innerHTML = items.map(([k, v]) => `<div class="event-item"><div class="name">${k}</div><div class="value">${fmt(v)}</div></div>`).join('');
}

function initStationSelects() {
  const options = state.data.stations.map((s) => `<option value="${s.station_id}">${s.station_name} (${s.station_id})</option>`).join('');
  const s1 = document.getElementById('stationSelect');
  const s2 = document.getElementById('replayStation');
  s1.innerHTML = options;
  s2.innerHTML = options;
  state.selectedStation = state.data.stations[0].station_id;
  s1.value = state.selectedStation;
  s2.value = state.selectedStation;

  s1.addEventListener('change', () => {
    state.selectedStation = s1.value;
    s2.value = s1.value;
    runPredictionDemo();
    renderReplay();
  });
  s2.addEventListener('change', () => {
    state.selectedStation = s2.value;
    s1.value = s2.value;
    renderReplay();
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
    xAxis: { type: 'category', data: forecast.map((x) => `${x.h}h`), axisLabel: { color: '#bfd4f6' } },
    yAxis: [
      { type: 'value', name: '风速', axisLabel: { color: '#bfd4f6' } },
      { type: 'value', name: '预警概率', min: 0, max: 1, axisLabel: { color: '#bfd4f6' } },
    ],
    series: [
      { name: '风速', type: 'line', smooth: true, data: forecast.map((x) => x.wind), lineStyle: { color: '#60a5fa' } },
      { name: '预警概率', type: 'bar', yAxisIndex: 1, data: forecast.map((x) => x.prob), itemStyle: { color: '#f59e0b' } },
    ],
  });

  document.getElementById('forecastTable').innerHTML = `
    <thead><tr><th>时效</th><th>风速</th><th>风速区间</th><th>风险等级</th><th>预警概率</th><th>概率区间</th></tr></thead>
    <tbody>
      ${forecast.map((x) => `<tr><td>${x.h}h</td><td>${x.wind}</td><td>${x.wind_p10} ~ ${x.wind_p90}</td><td style="color:${riskColor(x.risk)}">${riskLabel(x.risk)}</td><td>${x.prob}</td><td>${x.prob_p10} ~ ${x.prob_p90}</td></tr>`).join('')}
    </tbody>`;

  const driver = forecast[1] || forecast[0];
  const key = driver.risk >= 2 ? '风速升高 + 干燥增强 + 沙源接近度偏高' : '风速中等 + 湿度条件抑制 + 风险可控';
  document.getElementById('driverBox').innerHTML = `<strong>关键驱动因子说明：</strong>${key}`;
}

function mapValue(station, h, layer) {
  const p = stationProfile(station.station_id, h);
  if (layer === 'warn') return p.warn;
  if (layer === 'wind') return p.wind;
  return p.risk;
}

function updateMapDetail(stationId, h) {
  const s = state.data.stations.find((x) => x.station_id === stationId);
  const p = stationProfile(stationId, h);
  document.getElementById('detailName').textContent = s ? s.station_name : stationId;
  document.getElementById('detailHorizon').textContent = `${h}h`;
  document.getElementById('detailWind').textContent = `${fmt(p.wind)} m/s`;
  document.getElementById('detailRisk').textContent = riskLabel(p.risk);
  document.getElementById('detailRisk').style.color = riskColor(p.risk);
  document.getElementById('detailWarn').textContent = fmt(p.warn);
  document.getElementById('detailCI').textContent = `${fmt(p.wind_p10)} ~ ${fmt(p.wind_p90)} m/s`;
}

function renderMap() {
  const h = Number(state.mapHorizon);
  const layer = state.mapLayer;
  const chart = echarts.init(document.getElementById('mapChart'));
  const seriesData = state.data.stations.map((s) => {
    const v = mapValue(s, h, layer);
    const p = stationProfile(s.station_id, h);
    return {
      name: s.station_name,
      station_id: s.station_id,
      value: [s.lon, s.lat, v, p.risk, p.wind, p.warn],
    };
  });

  const tagText = layer === 'risk' ? `未来 ${h}h 风险态势` : layer === 'warn' ? `未来 ${h}h 预警概率` : `未来 ${h}h 风速预测`;
  document.getElementById('mapTag').textContent = tagText;

  const valueMinMax = layer === 'risk' ? [0, 3] : layer === 'warn' ? [0, 1] : [0, 18];
  const vmText = layer === 'risk' ? ['高', '低'] : layer === 'warn' ? ['高', '低'] : ['大', '小'];
  chart.setOption({
    backgroundColor: 'transparent',
    geo: {
      map: 'china',
      roam: true,
      center: [104.5, 38.2],
      zoom: 3.9,
      itemStyle: { areaColor: '#1b2d4c', borderColor: '#4f6992' },
      emphasis: { itemStyle: { areaColor: '#2b4772' } },
    },
    tooltip: {
      formatter: (p) => {
        const v = p.value || [];
        return `${p.name}<br/>风速:${fmt(v[4])} m/s<br/>风险:${riskLabel(v[3])}<br/>预警概率:${fmt(v[5])}`;
      },
    },
    visualMap: {
      min: valueMinMax[0],
      max: valueMinMax[1],
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      text: vmText,
      textStyle: { color: '#cde2ff' },
      calculable: false,
    },
    series: [
      {
        type: 'effectScatter',
        coordinateSystem: 'geo',
        data: seriesData,
        showEffectOn: 'render',
        rippleEffect: { scale: 3, brushType: 'stroke' },
        symbolSize: (v) => 10 + (layer === 'risk' ? v[2] * 5 : layer === 'warn' ? v[2] * 18 : v[2] * 0.9),
        itemStyle: {
          color: (p) => riskColor(p.data.value[3]),
          borderColor: '#fff',
          borderWidth: 1,
        },
      },
    ],
  });

  chart.off('click');
  chart.on('click', (params) => {
    if (params.data && params.data.station_id) updateMapDetail(params.data.station_id, h);
  });

  if (state.selectedStation) updateMapDetail(state.selectedStation, h);
}

function replayCurrentSlice() {
  const sid = state.selectedStation;
  const h = String(document.getElementById('replayHorizon').value);
  const rep = state.data.replay[sid][h];
  const caseIdx = Math.max(0, state.cases.findIndex((x) => x.date === state.selectedReplayDate));
  return sliceByCase(rep, caseIdx);
}

function renderReplay() {
  const rep = replayCurrentSlice();
  const slider = document.getElementById('replaySlider');
  slider.max = String(Math.max(0, rep.sample_idx.length - 1));
  slider.value = String(Math.min(Number(slider.value), rep.sample_idx.length - 1));

  const currentIdx = Number(slider.value);
  document.getElementById('replayHint').textContent = `当前时刻: T+${currentIdx}`;

  const x = rep.sample_idx.map((_, i) => i);

  const cw = echarts.init(document.getElementById('replayWindChart'));
  cw.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d8e8ff' } },
    xAxis: { type: 'category', data: x, axisLabel: { color: '#bdd4f6' } },
    yAxis: { type: 'value', axisLabel: { color: '#bdd4f6' } },
    series: [
      { name: '风速真值', type: 'line', data: rep.y_wind_true, lineStyle: { color: '#4ade80' } },
      { name: '风速预测', type: 'line', data: rep.y_wind_pred, lineStyle: { color: '#60a5fa' } },
      { name: '当前时刻', type: 'scatter', data: x.map((i) => (i === currentIdx ? rep.y_wind_pred[i] : null)), symbolSize: 12, itemStyle: { color: '#facc15' } },
    ],
  });

  const cp = echarts.init(document.getElementById('replayWarnChart'));
  cp.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#d8e8ff' } },
    xAxis: { type: 'category', data: x, axisLabel: { color: '#bdd4f6' } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#bdd4f6' } },
    series: [
      { name: '预警真值', type: 'line', step: 'middle', data: rep.y_warn_true, lineStyle: { color: '#f97316' } },
      { name: '预警概率', type: 'line', data: rep.y_warn_prob, lineStyle: { color: '#facc15' } },
      { name: '当前时刻', type: 'scatter', data: x.map((i) => (i === currentIdx ? rep.y_warn_prob[i] : null)), symbolSize: 12, itemStyle: { color: '#60a5fa' } },
    ],
  });
}

function runPredictionDemo() {
  const sid = state.selectedStation;
  const wind = Number(document.getElementById('windInput').value || 8);
  const rh = Number(document.getElementById('rhInput').value || 35);
  const f = simulateForecast(sid, wind, rh);
  renderForecast(f);
  renderMap();
}

function bindReplayControls() {
  const dateSel = document.getElementById('replayDate');
  const horizonSel = document.getElementById('replayHorizon');
  const slider = document.getElementById('replaySlider');

  dateSel.innerHTML = state.cases.map((c) => `<option value="${c.date}">${c.date} · ${c.label}</option>`).join('');
  state.selectedReplayDate = state.cases[0]?.date || null;
  dateSel.value = state.selectedReplayDate;

  dateSel.addEventListener('change', () => {
    state.selectedReplayDate = dateSel.value;
    slider.value = '0';
    renderReplay();
  });
  horizonSel.addEventListener('change', () => {
    slider.value = '0';
    renderReplay();
  });
  slider.addEventListener('input', renderReplay);

  document.getElementById('replayPlay').addEventListener('click', () => {
    if (state.replayTimer) return;
    state.replayTimer = setInterval(() => {
      const max = Number(slider.max || 0);
      let v = Number(slider.value || 0);
      v = v >= max ? 0 : v + 1;
      slider.value = String(v);
      renderReplay();
    }, 600);
  });
  document.getElementById('replayPause').addEventListener('click', () => {
    if (state.replayTimer) {
      clearInterval(state.replayTimer);
      state.replayTimer = null;
    }
  });
}

function bindMapControls() {
  const h = document.getElementById('mapHorizon');
  const l = document.getElementById('mapLayer');
  h.addEventListener('change', () => {
    state.mapHorizon = Number(h.value);
    renderMap();
  });
  l.addEventListener('change', () => {
    state.mapLayer = l.value;
    renderMap();
  });
}

function bindButtons() {
  document.getElementById('runBtn').addEventListener('click', runPredictionDemo);

  const apiBtn = document.getElementById('apiDocBtn');
  const appBtn = document.getElementById('onlineAppBtn');
  if (runtime.api_doc_url) apiBtn.href = runtime.api_doc_url;
  if (runtime.online_app_url) appBtn.href = runtime.online_app_url;
}

function onResize() {
  ['modelCompare', 'forecastChart', 'mapChart', 'replayWindChart', 'replayWarnChart'].forEach((id) => {
    const dom = document.getElementById(id);
    if (!dom) return;
    const inst = echarts.getInstanceByDom(dom);
    if (inst) inst.resize();
  });
}

(async function bootstrap() {
  await loadSiteConfig();
  const [data, cases] = await Promise.all([fetchJSON('./assets/demo_data.json'), fetchJSON('./assets/replay_cases.json')]);
  state.data = data;
  state.cases = cases;

  hydrateHero();
  renderModelCompare();
  renderEventSummary();
  initStationSelects();
  bindMapControls();
  bindReplayControls();
  bindButtons();

  runPredictionDemo();
  renderReplay();
  window.addEventListener('resize', onResize);
})();
