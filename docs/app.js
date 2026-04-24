const state = {
  data: null,
  selectedStation: null,
};

const riskColor = (r) => {
  if (r >= 3) return '#ef4444';
  if (r === 2) return '#f97316';
  if (r === 1) return '#f59e0b';
  return '#22c55e';
};

function fmt(n, d = 3) {
  return Number(n).toFixed(d);
}

function loadData() {
  return fetch('./assets/demo_data.json').then((r) => r.json());
}

function updateKpis() {
  const m = state.data.metrics.dustriskformer.regression;
  const e = state.data.event_summary.event_level_macro;
  document.getElementById('kpiMae').textContent = fmt(m.mae);
  document.getElementById('kpiRmse').textContent = fmt(m.rmse);
  document.getElementById('kpiR2').textContent = fmt(m.r2);
  document.getElementById('kpiHit').textContent = fmt(e.event_hit_rate);
}

function initStationSelector() {
  const sel = document.getElementById('stationSelect');
  sel.innerHTML = '';
  state.data.stations.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.station_id;
    opt.textContent = `${s.station_name} (${s.station_id})`;
    sel.appendChild(opt);
  });
  state.selectedStation = state.data.stations[0].station_id;
  sel.value = state.selectedStation;
  sel.addEventListener('change', () => {
    state.selectedStation = sel.value;
    runDemoPrediction();
  });
}

function simulateForecast(stationId, windInput, rhInput) {
  const base = state.data.base_profiles[stationId];
  const gain = state.data.sensitivity.wind_gain;
  const rhp = state.data.sensitivity.rh_penalty;

  const out = [];
  base.horizons.forEach((h, i) => {
    const baseWind = base.wind[i];
    const baseWarn = base.warn_prob[i];
    const adjWind = baseWind + gain[i] * (windInput - 8.0) - 0.03 * (rhInput - 35.0);
    const adjWarn = Math.min(0.99, Math.max(0.01, baseWarn + 0.035 * (windInput - 8.0) - rhp[i] * (rhInput - 35.0)));
    const risk = adjWarn > 0.8 ? 3 : adjWarn > 0.6 ? 2 : adjWarn > 0.35 ? 1 : 0;
    out.push({
      horizon: h,
      wind: Number(adjWind.toFixed(3)),
      wind_p10: Number((adjWind * 0.88).toFixed(3)),
      wind_p90: Number((adjWind * 1.12).toFixed(3)),
      warn: Number(adjWarn.toFixed(3)),
      warn_p10: Number(Math.max(0, adjWarn - 0.08).toFixed(3)),
      warn_p90: Number(Math.min(1, adjWarn + 0.08).toFixed(3)),
      risk,
    });
  });
  return out;
}

function renderForecast(forecast) {
  const c = echarts.init(document.getElementById('forecastChart'));
  c.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#dbe7ff' } },
    xAxis: { type: 'category', data: forecast.map((x) => `${x.horizon}h`), axisLabel: { color: '#b8c8e5' } },
    yAxis: [
      { type: 'value', name: '风速', axisLabel: { color: '#b8c8e5' } },
      { type: 'value', name: '预警概率', min: 0, max: 1, axisLabel: { color: '#b8c8e5' } },
    ],
    series: [
      { name: '风速预测', type: 'line', smooth: true, data: forecast.map((x) => x.wind) },
      { name: '预警概率', type: 'bar', yAxisIndex: 1, data: forecast.map((x) => x.warn), itemStyle: { color: '#f59e0b' } },
    ],
  });

  const table = document.getElementById('forecastTable');
  table.innerHTML = `
    <thead>
      <tr><th>时效</th><th>风速</th><th>区间(P10-P90)</th><th>风险</th><th>预警概率</th><th>区间(P10-P90)</th></tr>
    </thead>
    <tbody>
      ${forecast
        .map(
          (x) => `<tr>
            <td>${x.horizon}h</td>
            <td>${x.wind}</td>
            <td>${x.wind_p10} ~ ${x.wind_p90}</td>
            <td class="tag-risk-${x.risk}">${state.data.risk_labels[x.risk]}</td>
            <td>${x.warn}</td>
            <td>${x.warn_p10} ~ ${x.warn_p90}</td>
          </tr>`
        )
        .join('')}
    </tbody>
  `;
}

function renderMap() {
  const c = echarts.init(document.getElementById('mapChart'));
  const data = state.data.stations.map((s) => ({
    name: s.station_name,
    value: [s.lon, s.lat, s.warn_prob_3h, s.risk_3h],
    station_id: s.station_id,
  }));

  const baseOpt = {
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p) => {
        const v = p.value || [];
        return `${p.name}<br/>风险等级: ${state.data.risk_labels[v[3] || 0]}<br/>3h预警概率: ${Number(v[2] || 0).toFixed(3)}`;
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      textStyle: { color: '#dbe7ff' },
    },
    series: [
      {
        type: 'scatter',
        coordinateSystem: 'geo',
        symbolSize: (val) => 10 + val[2] * 18,
        data,
        itemStyle: {
          color: (p) => riskColor(p.data.value[3]),
          borderColor: '#fff',
          borderWidth: 1,
        },
      },
    ],
  };

  fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json')
    .then((r) => r.json())
    .then((geojson) => {
      echarts.registerMap('china', geojson);
      c.setOption({
        ...baseOpt,
        geo: {
          map: 'china',
          roam: true,
          itemStyle: { areaColor: '#21304f', borderColor: '#4b5f87' },
          emphasis: { itemStyle: { areaColor: '#2f436d' } },
        },
      });
    })
    .catch(() => {
      c.setOption({
        backgroundColor: 'transparent',
        tooltip: baseOpt.tooltip,
        xAxis: { type: 'value', name: '经度', axisLabel: { color: '#b8c8e5' } },
        yAxis: { type: 'value', name: '纬度', axisLabel: { color: '#b8c8e5' } },
        series: [
          {
            type: 'scatter',
            data: data.map((d) => ({ name: d.name, value: [d.value[0], d.value[1], d.value[2], d.value[3]] })),
            symbolSize: (val) => 10 + val[2] * 18,
            itemStyle: { color: (p) => riskColor(p.data.value[3]) },
          },
        ],
      });
    });
}

function renderReplay(stationId) {
  const rep = state.data.replay[stationId]['3'];

  const c1 = echarts.init(document.getElementById('replayWindChart'));
  c1.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: rep.sample_idx, axisLabel: { color: '#b8c8e5', show: false } },
    yAxis: { type: 'value', axisLabel: { color: '#b8c8e5' } },
    legend: { textStyle: { color: '#dbe7ff' } },
    series: [
      { name: '真值', type: 'line', data: rep.y_wind_true, lineStyle: { color: '#4ade80' } },
      { name: '预测', type: 'line', data: rep.y_wind_pred, lineStyle: { color: '#60a5fa' } },
    ],
  });

  const c2 = echarts.init(document.getElementById('replayWarnChart'));
  c2.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: rep.sample_idx, axisLabel: { color: '#b8c8e5', show: false } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#b8c8e5' } },
    legend: { textStyle: { color: '#dbe7ff' } },
    series: [
      { name: '预警真值', type: 'line', step: 'middle', data: rep.y_warn_true, lineStyle: { color: '#f97316' } },
      { name: '预警概率', type: 'line', data: rep.y_warn_prob, lineStyle: { color: '#facc15' } },
    ],
  });
}

function runDemoPrediction() {
  const sid = state.selectedStation;
  const wind = Number(document.getElementById('windInput').value || 8);
  const rh = Number(document.getElementById('rhInput').value || 35);
  const forecast = simulateForecast(sid, wind, rh);
  renderForecast(forecast);
  renderReplay(sid);
}

async function bootstrap() {
  state.data = await loadData();
  updateKpis();
  initStationSelector();
  renderMap();

  document.getElementById('runBtn').addEventListener('click', runDemoPrediction);
  runDemoPrediction();

  window.addEventListener('resize', () => {
    ['forecastChart', 'mapChart', 'replayWindChart', 'replayWarnChart'].forEach((id) => {
      const instance = echarts.getInstanceByDom(document.getElementById(id));
      if (instance) instance.resize();
    });
  });
}

bootstrap();
