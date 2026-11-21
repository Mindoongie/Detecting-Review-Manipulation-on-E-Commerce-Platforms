import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { normalize, timestamp } from '../src/temporal.mjs';
import { configuration, normalizeGroups, groupScores, burstScore, stage2 } from '../src/group-temporal.mjs';

const r = (id, user, day, rating = 5, product = 'p') => ({ review_id: id, user_id: user, product_id: product, timestamp: `2024-01-${day}`, rating });
const g = (extra = {}) => ({ group_id: 'g', user_ids: ['a','b'], product_ids: ['p'], mined_through: '2024-01-03', ...extra });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
const base = [r('o', 'other', '01', 1), r('a', 'a', '02'), r('b', 'b', '03'), r('target', 'a', '04')];

test('group metrics match hand calculation; GETF uses latest member timestamp', () => {
  const score = groupScores(normalize(base.slice(0, 3)), normalizeGroups([g()])[0], { tau_days: 10, beta_days: 10 });
  close(score.gtw, 0.9); close(score.getf, 0.8); close(score.gd, 1);
  assert.equal(score.gd_missing, 0);
});
test('no outsider, empty support, singleton evidence and expired windows', () => {
  const candidate = normalizeGroups([g()])[0];
  const score = groupScores(normalize(base.slice(1, 3)), candidate, { tau_days: 0.5, beta_days: 0.5 });
  assert.equal(score.gd, 0); assert.equal(score.gd_missing, 1);
  assert.equal(score.gtw, 0); assert.equal(score.getf, 0);
  assert.equal(groupScores(normalize([base[1]]), candidate).no_group_evidence, 1);
  assert.equal(groupScores(normalize(base), { ...candidate, product_ids: [] }).no_group_evidence, 1);
});
test('group max aggregates across products and ignores products with missing GD', () => {
  const history = normalize([...base.slice(0, 3), r('qa','a','01',5,'q'), r('qb','b','01',5,'q')]);
  const score = groupScores(history, normalizeGroups([g({ product_ids: ['p','q'] })])[0], { tau_days: 10, beta_days: 10 });
  assert.equal(score.eligible_products, 2);
  assert.equal(score.gtw, 1); assert.equal(score.getf, 1); assert.equal(score.gd, 1);
  assert.equal(score.gd_missing, 0);
  assert.equal(score.details.find(d => d.product_id === 'q').gd_missing, 1);
});
test('latest review per user-product; ties deterministic; availability retains earliest observation', () => {
  const history = [...base.slice(0, 3), r('z', 'a', '05', 1), r('y', 'a', '05', 3)];
  const candidate = normalizeGroups([g()])[0];
  const score = groupScores(normalize(history), candidate, { tau_days: 10, beta_days: 10 });
  close(score.gd, 0.5); close(score.gtw, 0.8); close(score.getf, 0.6);
  assert.deepEqual(score, groupScores(normalize(history.reverse()), candidate, { tau_days: 10, beta_days: 10 }));
});
test('groups only apply after mining cutoff and on supporting product; all review rows retained', () => {
  const result = stage2([...base, r('other-product', 'a', '05', 5, 'q')], [g()], '2024-01-15');
  assert.equal(result.reviewFeatures.length, 5);
  assert.equal(result.reviewFeatures.find(r => r.review_id === 'b').no_group_evidence, 1);
  assert.equal(result.reviewFeatures.find(r => r.review_id === 'target').group_count, 1);
  assert.equal(result.reviewFeatures.find(r => r.review_id === 'other-product').no_group_evidence, 1);
});
test('future reviews, future candidates, labels and same-time peers cannot change current features', () => {
  const original = stage2(base, [g()], '2024-01-15');
  const futureGroup = g({ group_id: 'future', mined_through: '2024-02-01' });
  const changed = stage2([...base.map(x => ({ ...x, label: 1 })), { ...r('future', 'a', '30'), timestamp: '2024-02-01' }], [g(), futureGroup], '2024-01-15');
  assert.deepEqual(original.reviewFeatures, changed.reviewFeatures);
  assert.deepEqual(original.groupFeatures, changed.groupFeatures);
  const peers = stage2([...base, r('peer', 'a', '04', 1)], [g()], '2024-01-15');
  assert.deepEqual(original.reviewFeatures.find(r => r.review_id === 'target'), peers.reviewFeatures.find(r => r.review_id === 'target'));
});
test('invalid mining support fails; newer empty version retires prior evidence', () => {
  assert.throws(() => stage2(base, [g({ mined_through: '2024-01-01' })], '2024-01-15'), /lacks member evidence/);
  const result = stage2([...base, r('next', 'a', '05')], [g(), g({ mined_through: '2024-01-04', product_ids: [] })], '2024-01-15');
  assert.equal(result.reviewFeatures.find(r => r.review_id === 'target').group_count, 1);
  assert.equal(result.reviewFeatures.find(r => r.review_id === 'next').group_count, 0);
});
test('overlapping group aggregation omits missing GD from its mean', () => {
  const all = g({ group_id: 'all', user_ids: ['a','b','other'] });
  const result = stage2(base, [g(), all], '2024-01-15').reviewFeatures.find(r => r.review_id === 'target');
  assert.equal(result.group_count, 2); assert.equal(result.gd_valid_group_count, 1);
  assert.equal(result.gd_mean, 1); assert.equal(result.gd_missing, 0);
});
test('burst uses disjoint past windows with exact boundaries and finite zero-variance behavior', () => {
  const history = normalize([r('start','x','01'), r('baseline','x','08'), r('s1','x','09'), r('s2','y','09'), r('s3','z','09'), r('same','z','10')]);
  const score = burstScore(history, 'p', timestamp('2024-01-10'));
  assert.equal(score.burst_recent_count, 3); close(score.burst_baseline_mean, 1 / 7);
  assert.equal(score.burst_history_missing, 0); assert.equal(score.burst_flag, 1);
  const zero = burstScore(normalize([r('start','x','01'), r('s1','x','09'), r('s2','y','09'), r('s3','z','09')]), 'p', timestamp('2024-01-10'));
  assert.equal(zero.burst_baseline_std, 0); assert.equal(zero.burst_flag, 1);
  const cold = burstScore(normalize([r('s1','x','09'), r('s2','y','09'), r('s3','z','09')]), 'p', timestamp('2024-01-10'));
  assert.equal(cold.burst_history_missing, 1); assert.equal(cold.burst_flag, 0);
});
test('schema and configuration reject duplicates, malformed groups and invalid parameters', () => {
  assert.throws(() => normalizeGroups([g(), g()]), /Duplicate/);
  for (const candidate of [g({ user_ids: ['a','a'] }), g({ user_ids: ['a'] }), g({ mined_through: undefined }), g({ product_ids: ['p','p'] })]) assert.throws(() => normalizeGroups([candidate]));
  for (const c of [{ tau_days: 0 }, { beta_days: -1 }, { baseline_windows: 1.5 }, { burst_window_days: NaN }, { nonsense: 1 }]) assert.throws(() => configuration(c));
});
test('empty groups and no observed reviews return clean results', () => {
  assert.ok(stage2(base, [], '2024-01-15').reviewFeatures.every(r => r.no_group_evidence && r.gd_missing));
  assert.equal(stage2(base, [], '2023-01-01').reviewFeatures.length, 0);
});
test('CLI produces mock-tagged summary and header-only tables when groups are empty', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stage2-test-'));
  fs.writeFileSync(path.join(dir, 'groups.jsonl'), '');
  execFileSync(process.execPath, ['src/group-temporal.mjs', '--input', 'data/mock/stage2_reviews.jsonl', '--groups', path.join(dir, 'groups.jsonl'), '--cutoff', '2024-01-15', '--out', dir, '--mock']);
  const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
  assert.match(summary.data_kind, /MOCK/);
  assert.equal(summary.observed_reviews, 6);
  assert.equal(fs.readFileSync(path.join(dir, 'group_temporal_features.csv'), 'utf8').trim().split('\n').length, 1);
  // Remove only the exact files created by this test, without recursive deletion.
  for (const name of ['groups.jsonl','summary.json','group_temporal_features.csv','group_product_features.csv','review_temporal_group_features.csv']) fs.unlinkSync(path.join(dir, name));
  fs.rmdirSync(dir);
});
