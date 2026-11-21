import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { normalize, timestamp } from './temporal.mjs';

const DAY = 86400000;
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
const maximum = xs => xs.reduce((a, b) => Math.max(a, b), 0);
export const defaults = Object.freeze({ tau_days: 30, beta_days: 180, burst_window_days: 1, baseline_windows: 7, burst_min_count: 3, burst_std_multiplier: 3 });

export function configuration(input = {}) {
  for (const key of Object.keys(input)) if (!(key in defaults)) throw new Error(`Unknown config: ${key}`);
  const c = { ...defaults, ...input };
  for (const [key, value] of Object.entries(c)) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) throw new Error(`Config ${key} must be positive and finite`);
  }
  for (const key of ['baseline_windows', 'burst_min_count']) if (!Number.isInteger(c[key])) throw new Error(`${key} must be an integer`);
  return c;
}

export function normalizeGroups(records) {
  const seen = new Set();
  return records.map(g => {
    if (typeof g.group_id !== 'string' || !g.group_id.trim()) throw new Error('Missing group_id');
    for (const key of ['user_ids', 'product_ids']) {
      if (!Array.isArray(g[key]) || g[key].some(x => typeof x !== 'string' || !x.trim()) || new Set(g[key]).size !== g[key].length) throw new Error(`Invalid ${key} for ${g.group_id}`);
    }
    if (g.user_ids.length < 2) throw new Error('A candidate group needs at least two users');
    const minedThrough = timestamp(g.mined_through);
    const key = JSON.stringify([g.group_id, minedThrough]);
    if (seen.has(key)) throw new Error(`Duplicate group version: ${key}`);
    seen.add(key);
    return { group_id: g.group_id, user_ids: [...g.user_ids], product_ids: [...g.product_ids], mined_through: new Date(minedThrough).toISOString(), minedThrough };
  });
}

// Inclusive declared mining cutoff must precede a review's decision timestamp.
function activeGroups(groups, decision) {
  const latest = new Map();
  for (const g of groups) {
    if (g.minedThrough < decision && (!latest.has(g.group_id) || latest.get(g.group_id).minedThrough < g.minedThrough)) latest.set(g.group_id, g);
  }
  return [...latest.values()].sort((a, b) => a.group_id.localeCompare(b.group_id));
}

export function groupScores(history, group, config = {}) {
  const c = configuration(config), members = new Set(group.user_ids), details = [];
  for (const product of group.product_ids) {
    const rows = history.filter(r => r.product_id === product);
    const latest = new Map();
    for (const r of rows) {
      const old = latest.get(r.user_id);
      if (!old || r.time > old.time || (r.time === old.time && r.review_id > old.review_id)) latest.set(r.user_id, r);
    }
    const inside = [...latest.values()].filter(r => members.has(r.user_id));
    if (inside.length < 2) continue;
    const outside = [...latest.values()].filter(r => !members.has(r.user_id));
    const first = inside.reduce((t, r) => Math.min(t, r.time), Infinity);
    const last = inside.reduce((t, r) => Math.max(t, r.time), -Infinity);
    const available = rows.reduce((t, r) => Math.min(t, r.time), Infinity);
    details.push({ group_id: group.group_id, product_id: product, member_count: inside.length,
      outside_count: outside.length, gtw: Math.max(0, 1 - (last - first) / DAY / c.tau_days),
      gd: outside.length ? Math.abs(mean(inside.map(r => r.rating)) - mean(outside.map(r => r.rating))) / 4 : 0,
      gd_missing: Number(!outside.length), getf: Math.max(0, 1 - (last - available) / DAY / c.beta_days), availability_is_proxy: 1 });
  }
  const validGD = details.filter(d => !d.gd_missing);
  return { group_id: group.group_id, mined_through: group.mined_through,
    eligible_products: details.length, gtw: maximum(details.map(d => d.gtw)),
    gd: maximum(validGD.map(d => d.gd)), getf: maximum(details.map(d => d.getf)),
    gd_missing: Number(!validGD.length), no_group_evidence: Number(!details.length),
    availability_is_proxy: 1, details };
}

export function burstScore(history, product, decision, config = {}) {
  const c = configuration(config), width = c.burst_window_days * DAY;
  const times = history.filter(r => r.product_id === product && r.time < decision).map(r => r.time);
  const count = (lo, hi) => times.filter(t => t >= lo && t < hi).length;
  const current = count(decision - width, decision);
  const baseline = Array.from({ length: c.baseline_windows }, (_, i) => count(decision - (i + 2) * width, decision - (i + 1) * width));
  const avg = mean(baseline), std = Math.sqrt(mean(baseline.map(n => (n - avg) ** 2)));
  const first = times.reduce((t, x) => Math.min(t, x), Infinity);
  const ready = first <= decision - (c.baseline_windows + 1) * width;
  return { burst_recent_count: current, burst_baseline_mean: avg, burst_baseline_std: std,
    burst_history_missing: Number(!ready), burst_excess: Math.max(0, current - avg),
    burst_flag: Number(ready && current >= c.burst_min_count && current > avg + c.burst_std_multiplier * std) };
}

export function stage2(rawReviews, rawGroups, cutoffISO, config = {}) {
  const c = configuration(config), cutoff = timestamp(cutoffISO), rows = normalize(rawReviews), groups = normalizeGroups(rawGroups);
  const observed = rows.filter(r => r.time <= cutoff), reviewFeatures = [], groupFeatures = [], productFeatures = [], history = [];
  for (let start = 0; start < observed.length;) {
    const decision = observed[start].time, decision_time = observed[start].timestamp;
    let end = start + 1;
    while (end < observed.length && observed[end].time === decision) end++;
    const active = activeGroups(groups, decision);
    // Validate declared membership/support using only evidence available at mining time.
    for (const g of active) {
      for (const product of g.product_ids) {
        const users = new Set(history.filter(r => r.time <= g.minedThrough && r.product_id === product).map(r => r.user_id));
        if (!g.user_ids.every(u => users.has(u))) throw new Error(`Group ${g.group_id}: product ${product} lacks member evidence at mined_through`);
      }
    }
    const scored = active.map(g => ({ candidate: g, score: groupScores(history, g, c) }));
    for (const { score } of scored) {
      const { details, ...summary } = score;
      groupFeatures.push({ decision_time, ...summary });
      productFeatures.push(...details.map(d => ({ decision_time, ...d })));
    }
    const bursts = new Map();
    for (let i = start; i < end; i++) {
      const r = observed[i];
      if (!bursts.has(r.product_id)) bursts.set(r.product_id, burstScore(history, r.product_id, decision, c));
      const assigned = scored.filter(({ candidate, score }) => candidate.user_ids.includes(r.user_id) && score.details.some(d => d.product_id === r.product_id)).map(x => x.score);
      const validGD = assigned.filter(g => !g.gd_missing);
      const feature = { review_id: r.review_id, user_id: r.user_id, product_id: r.product_id, decision_time,
        snapshot_cutoff: new Date(cutoff).toISOString(), ...bursts.get(r.product_id),
        group_count: assigned.length, no_group_evidence: Number(!assigned.length),
        gd_valid_group_count: validGD.length, gd_missing: Number(!validGD.length) };
      for (const name of ['gtw', 'gd', 'getf']) {
        const values = (name === 'gd' ? validGD : assigned).map(g => g[name]);
        feature[`${name}_max`] = maximum(values); feature[`${name}_mean`] = mean(values);
      }
      reviewFeatures.push(feature);
    }
    for (let i = start; i < end; i++) history.push(observed[i]);
    start = end;
  }
  return { reviewFeatures, groupFeatures, productFeatures,
    summary: { input_reviews: rows.length, observed_reviews: observed.length, excluded_future_reviews: rows.length - observed.length,
      candidate_versions: groups.length, review_count_with_group: reviewFeatures.filter(r => r.group_count).length,
      burst_review_count: reviewFeatures.filter(r => r.burst_flag).length, config: c,
      cutoff: new Date(cutoff).toISOString(), protocol: 'history < decision_time; mined_through < decision_time; review timestamp <= cutoff',
      availability_is_proxy: true, gd_variant: 'omit products without outside reviewers; zero + missing flag if none' } };
}

const reviewColumns = ['review_id','user_id','product_id','decision_time','snapshot_cutoff','burst_recent_count','burst_baseline_mean','burst_baseline_std','burst_history_missing','burst_excess','burst_flag','group_count','no_group_evidence','gd_valid_group_count','gd_missing','gtw_max','gtw_mean','gd_max','gd_mean','getf_max','getf_mean'];
const groupColumns = ['decision_time','group_id','mined_through','eligible_products','gtw','gd','getf','gd_missing','no_group_evidence','availability_is_proxy'];
const productColumns = ['decision_time','group_id','product_id','member_count','outside_count','gtw','gd','gd_missing','getf','availability_is_proxy'];
function writeCSV(file, rows, columns) {
  const escape = x => `"${String(x ?? '').replaceAll('"', '""')}"`;
  fs.writeFileSync(file, [columns.map(escape).join(','), ...rows.map(r => columns.map(k => escape(r[k])).join(','))].join('\n') + '\n');
}
function readJSONL(file) {
  return fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, '').split(/\r?\n/).filter(s => s.trim()).map(line => JSON.parse(line));
}
function main() {
  const options = {}, args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--mock') options.mock = true;
    else if (['--input','--groups','--cutoff','--config','--out'].includes(args[i]) && args[i + 1] && !args[i + 1].startsWith('--')) options[args[i].slice(2)] = args[++i];
    else throw new Error(`Unknown/incomplete argument: ${args[i]}`);
  }
  for (const key of ['input','groups','cutoff','out']) if (!options[key]) throw new Error(`Missing --${key}`);
  const result = stage2(readJSONL(options.input), readJSONL(options.groups), options.cutoff, options.config ? JSON.parse(fs.readFileSync(options.config, 'utf8').replace(/^\uFEFF/, '')) : {});
  fs.mkdirSync(options.out, { recursive: true });
  writeCSV(path.join(options.out, 'review_temporal_group_features.csv'), result.reviewFeatures, reviewColumns);
  writeCSV(path.join(options.out, 'group_temporal_features.csv'), result.groupFeatures, groupColumns);
  writeCSV(path.join(options.out, 'group_product_features.csv'), result.productFeatures, productColumns);
  result.summary.data_kind = options.mock ? 'MOCK - NOT REAL DATASET RESULTS' : 'USER PROVIDED - mining provenance must be verified by TV2';
  fs.writeFileSync(path.join(options.out, 'summary.json'), JSON.stringify(result.summary, null, 2));
  console.log(JSON.stringify(result.summary, null, 2));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
