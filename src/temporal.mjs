import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DAY = 86400000;
const iso = ms => new Date(ms).toISOString();
const date = ms => iso(ms).slice(0, 10);
const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;

// Date-only values are explicitly interpreted as UTC, never as the host timezone.
export function timestamp(value) {
  if (typeof value !== 'string') throw new Error('timestamp must be an ISO string');
  const match = /^(\d{4}-\d{2}-\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(Z|[+-]\d{2}:\d{2}))?$/.exec(value);
  if (!match) throw new Error(`Invalid or timezone-ambiguous timestamp: ${value}`);
  const calendar = new Date(`${match[1]}T00:00:00Z`);
  if (!Number.isFinite(+calendar) || date(+calendar) !== match[1]) throw new Error(`Invalid calendar date: ${value}`);
  if (match[2] && (+match[2] > 23 || +match[3] > 59 || +match[4] > 59)) throw new Error(`Invalid clock: ${value}`);
  const result = Date.parse(value.length === 10 ? `${value}T00:00:00Z` : value);
  if (!Number.isFinite(result)) throw new Error(`Invalid timestamp: ${value}`);
  return result;
}

export function normalize(records) {
  const seen = new Set();
  return records.map((r, i) => {
    for (const key of ['review_id', 'user_id', 'product_id']) {
      if (typeof r[key] !== 'string' || !r[key].trim()) throw new Error(`Row ${i + 1}: missing string ${key}`);
    }
    if (seen.has(r.review_id)) throw new Error(`Duplicate review_id: ${r.review_id}`);
    seen.add(r.review_id);
    if (typeof r.rating !== 'number' || !Number.isFinite(r.rating) || r.rating < 1 || r.rating > 5) throw new Error(`Row ${i + 1}: rating must be numeric in [1,5]`);
    const time = timestamp(r.timestamp);
    // Whitelist prevents labels and arbitrary metadata from entering features.
    return { review_id: r.review_id, user_id: r.user_id, product_id: r.product_id, timestamp: iso(time), rating: r.rating, time };
  }).sort((a, b) => a.time - b.time || a.review_id.localeCompare(b.review_id));
}

function state() { return { count: 0, sum: 0, sumsq: 0, first: null, last: null, products: new Set(), times: [], head: 0 }; }
function update(s, r) {
  s.count++; s.sum += r.rating; s.sumsq += r.rating ** 2;
  s.first ??= r.time; s.last = r.time; s.products.add(r.product_id); s.times.push(r.time);
}
function recent(s, time) {
  while (s.head < s.times.length && s.times[s.head] < time - 7 * DAY) s.head++;
  return s.times.length - s.head;
}

export function features(rows, cutoff) {
  const observed = rows.filter(r => r.time <= cutoff).slice().sort((a, b) => a.time - b.time || a.review_id.localeCompare(b.review_id));
  const users = new Map(), products = new Map(), result = [];
  for (let start = 0; start < observed.length;) {
    let end = start + 1;
    while (end < observed.length && observed[end].time === observed[start].time) end++;
    // Evaluate the whole timestamp batch before updating history: no arbitrary tie order.
    for (let i = start; i < end; i++) {
      const r = observed[i], u = users.get(r.user_id) ?? state(), p = products.get(r.product_id) ?? state();
      const userMean = u.count ? u.sum / u.count : 0;
      const productMean = p.count ? p.sum / p.count : 0;
      result.push({ review_id: r.review_id, user_id: r.user_id, product_id: r.product_id,
        decision_time: r.timestamp, snapshot_cutoff: iso(cutoff), rating: r.rating,
        user_prior_count: u.count, user_prior_products: u.products.size,
        user_prior_7d_count: recent(u, r.time), user_history_missing: Number(!u.count),
        user_observed_age_days: u.count ? (r.time - u.first) / DAY : 0,
        user_gap_days: u.count ? (r.time - u.last) / DAY : 0,
        user_prior_rating_mean: userMean,
        user_prior_rating_std: u.count ? Math.sqrt(Math.max(0, u.sumsq / u.count - userMean ** 2)) : 0,
        product_prior_count: p.count, product_prior_7d_count: recent(p, r.time),
        product_history_missing: Number(!p.count), product_prior_rating_mean: productMean,
        product_observed_age_days: p.count ? (r.time - p.first) / DAY : 0,
        rating_deviation_prior_product: p.count ? Math.abs(r.rating - productMean) / 4 : 0 });
    }
    for (let i = start; i < end; i++) {
      const r = observed[i];
      if (!users.has(r.user_id)) users.set(r.user_id, state());
      if (!products.has(r.product_id)) products.set(r.product_id, state());
      update(users.get(r.user_id), r); update(products.get(r.product_id), r);
    }
    start = end;
  }
  return result;
}

export function eda(rows, cutoff) {
  const observed = rows.filter(r => r.time <= cutoff);
  if (!observed.length) throw new Error('No reviews at or before cutoff');
  const counts = new Map(), byProduct = new Map();
  for (const r of observed) {
    const d = date(r.time);
    counts.set(d, (counts.get(d) ?? 0) + 1);
    if (!byProduct.has(r.product_id)) byProduct.set(r.product_id, new Map());
    const c = byProduct.get(r.product_id); c.set(d, (c.get(d) ?? 0) + 1);
  }
  const first = observed.reduce((earliest, r) => Math.min(earliest, r.time), Infinity);
  const daily = [], weekly = new Map(), productDaily = [];
  const firstDay = timestamp(date(first));
  for (let t = firstDay; t <= cutoff; t += DAY) {
    const d = date(t), n = counts.get(d) ?? 0;
    daily.push({ date: d, review_count: n });
    const monday = date(t - ((new Date(t).getUTCDay() + 6) % 7) * DAY);
    weekly.set(monday, (weekly.get(monday) ?? 0) + n);
  }
  for (const [product, c] of byProduct) {
    const firstProductDay = [...c.keys()].sort()[0];
    for (let t = timestamp(firstProductDay); t <= cutoff; t += DAY) {
      const d = date(t), count = c.get(d) ?? 0;
      const available = Math.round((t - timestamp(firstProductDay)) / DAY);
      const history = Array.from({ length: Math.min(7, available) }, (_, i) => c.get(date(t - (i + 1) * DAY)) ?? 0);
      const avg = mean(history), std = Math.sqrt(mean(history.map(x => (x - avg) ** 2)));
      productDaily.push({ product_id: product, date: d, review_count: count,
        baseline_days: history.length, prior_7d_daily_mean: avg,
        spike_candidate: Number(history.length === 7 && count >= 3 && count > avg + 3 * std) });
    }
  }
  return { daily, weekly: [...weekly].map(([week_start, review_count]) => ({ week_start, review_count })), productDaily };
}

function csv(rows) {
  if (!rows.length) return '';
  const keys = Object.keys(rows[0]);
  const escape = x => `"${String(x ?? '').replaceAll('"', '""')}"`;
  return [keys.map(escape).join(','), ...rows.map(r => keys.map(k => escape(r[k])).join(','))].join('\n') + '\n';
}

function report(tables, summary) {
  const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  const bars = (rows, key) => {
    const max = rows.reduce((n, r) => Math.max(n, r.review_count), 1);
    return rows.map(r => `<div class="row"><time>${escape(r[key])}</time><div class="track"><div class="bar" style="width:${100 * r.review_count / max}%"></div></div><b>${r.review_count}</b></div>`).join('');
  };
  return `<!doctype html><html lang="vi"><meta charset="utf-8"><title>EDA thời gian — TV4</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#172b42}h1{font-size:28px}.notice{background:#fff1cf;padding:16px}.row{display:flex;gap:16px;align-items:center;margin:5px 0}time{width:110px}.track{flex:1;background:#edf2f7;height:16px}.bar{background:#226b99;height:100%}b{width:48px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style>
<h1>EDA thời gian — Giai đoạn 1 / Hiền</h1><p class="notice">${escape(summary.data_kind)}</p>
<p>Mốc cắt UTC: ${escape(summary.cutoff)}. Có ${summary.observed_reviews} review được quan sát; loại ${summary.excluded_future_reviews} review tương lai.</p>
<h2>Số review theo ngày (UTC)</h2>${bars(tables.daily, 'date')}
<h2>Số review theo tuần (bắt đầu thứ Hai, UTC)</h2>${bars(tables.weekly, 'week_start')}
<h2>Ngày có lượng review tăng mạnh trên sản phẩm</h2><p>Điều kiện thăm dò: có đủ 7 ngày nền, số review ≥ 3 và lớn hơn trung bình 7 ngày trước + 3 độ lệch chuẩn. Đây không phải kết luận thao túng.</p>
<table><tr><th>Sản phẩm</th><th>Ngày</th><th>Số review</th><th>Trung bình 7 ngày trước</th></tr>${summary.spike_candidates.map(r => `<tr><td>${escape(r.product_id)}</td><td>${r.date}</td><td>${r.review_count}</td><td>${r.prior_7d_daily_mean.toFixed(3)}</td></tr>`).join('')}</table>
<p>Ngày cuối/tuần đầu và cuối có thể chưa đủ kỳ. Các ngày không có review đã được điền 0, chỉ từ ngày sản phẩm xuất hiện. Khi chạy dữ liệu thật, chỉ EDA trên train.</p></html>`;
}

function main() {
  const args = process.argv.slice(2), options = {};
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--mock') options.mock = true;
    else if (['--input', '--cutoff', '--out'].includes(arg) && args[i + 1] && !args[i + 1].startsWith('--')) options[arg.slice(2)] = args[++i];
    else throw new Error(`Unknown/incomplete argument: ${arg}`);
  }
  if (!options.input || !options.cutoff || !options.out) throw new Error('Usage: node src/temporal.mjs --input reviews.jsonl --cutoff ISO --out directory [--mock]');
  const cutoff = timestamp(options.cutoff);
  const rows = normalize(fs.readFileSync(options.input, 'utf8').replace(/^\uFEFF/, '').split(/\r?\n/).filter(x => x.trim()).map((line, i) => {
    try { return JSON.parse(line); } catch { throw new Error(`Invalid JSON at line ${i + 1}`); }
  }));
  const observed = rows.filter(r => r.time <= cutoff), tables = eda(rows, cutoff), extracted = features(rows, cutoff);
  fs.mkdirSync(options.out, { recursive: true });
  for (const [name, table] of Object.entries({ normalized_reviews: observed.map(({ time, ...r }) => r), b1_temporal_features: extracted, daily_counts: tables.daily, weekly_counts: tables.weekly, product_daily_counts: tables.productDaily })) {
    fs.writeFileSync(path.join(options.out, `${name}.csv`), csv(table));
  }
  const summary = { data_kind: options.mock ? 'MOCK - NOT REAL DATASET RESULTS' : 'USER PROVIDED - provenance not verified',
    cutoff: iso(cutoff), input_reviews: rows.length, observed_reviews: observed.length,
    excluded_future_reviews: rows.length - observed.length, users: new Set(observed.map(r => r.user_id)).size,
    products: new Set(observed.map(r => r.product_id)).size,
    timestamp_policy: 'UTC; date-only=00:00 UTC; timezone-less datetime rejected',
    feature_policy: 'history timestamp < decision_time <= cutoff; ties excluded from history',
    spike_candidates: tables.productDaily.filter(r => r.spike_candidate),
    limitations: ['No spam inference from spikes', 'Partial first/last weeks and final day possible', 'Observed age is not account/product creation age', 'EDA must use train cutoff; validation/test remain held out'] };
  fs.writeFileSync(path.join(options.out, 'summary.json'), JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(options.out, 'eda.html'), report(tables, summary));
  console.log(JSON.stringify(summary, null, 2));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
