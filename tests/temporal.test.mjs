import test from 'node:test';
import assert from 'node:assert/strict';
import { normalize, timestamp, features, eda } from '../src/temporal.mjs';

const row = (id, user, day, rating = 5, product = 'p1') => ({ review_id: id, user_id: user, product_id: product, timestamp: `2024-01-${day}`, rating });
const cutoff = timestamp('2024-01-15');

test('timestamps: UTC, equivalent offset, invalid dates, missing timezone and epoch', () => {
  assert.equal(timestamp('2024-01-01'), timestamp('2024-01-01T07:00:00+07:00'));
  for (const input of ['2024-02-30', '2024-01-01T00:00:00', '2024-01-01T24:00:00Z', 1704067200]) assert.throws(() => timestamp(input));
});
test('schema rejects duplicate IDs and invalid/missing ratings, preserves all users', () => {
  const r = row('r1', 'u1', '01');
  assert.throws(() => normalize([r, r]), /Duplicate/);
  for (const rating of [undefined, null, '5', NaN, 6, 0]) assert.throws(() => normalize([{ ...r, rating }]));
  assert.equal(features(normalize([r]), cutoff).length, 1);
});
test('hand-calculated features use strictly earlier history; cold starts are flagged', () => {
  const result = features(normalize([row('a', 'u1', '01', 1), row('b', 'u2', '02', 3), row('c', 'u1', '08', 5)]), cutoff);
  assert.equal(result[0].user_history_missing, 1);
  assert.equal(result[0].product_history_missing, 1);
  assert.equal(result[2].user_prior_count, 1);
  assert.equal(result[2].user_prior_7d_count, 1); // inclusive lower boundary
  assert.equal(result[2].user_gap_days, 7);
  assert.equal(result[2].product_prior_rating_mean, 2);
  assert.equal(result[2].rating_deviation_prior_product, 0.75);
});
test('same timestamp reviews cannot see each other and input order is irrelevant', () => {
  const rows = [row('a', 'u1', '01'), row('b', 'u1', '01')];
  const a = features(normalize(rows), cutoff), b = features(normalize([...rows].reverse()), cutoff);
  assert.deepEqual(a, b);
  assert.ok(a.every(r => r.user_prior_count === 0 && r.product_prior_count === 0));
});
test('future data and labels do not affect prior features or EDA', () => {
  const base = [row('a', 'u1', '01'), row('b', 'u1', '10')];
  const augmented = [...base.map(r => ({ ...r, label: 1 })), row('future', 'u1', '30', 1)];
  assert.deepEqual(features(normalize(base), cutoff), features(normalize(augmented), cutoff));
  assert.deepEqual(eda(normalize(base), cutoff), eda(normalize(augmented), cutoff));
  const early = features(normalize(base), timestamp('2024-01-05'))[0];
  const late = features(normalize(base), cutoff)[0];
  const { snapshot_cutoff: ignored1, ...a } = early;
  const { snapshot_cutoff: ignored2, ...b } = late;
  assert.deepEqual(a, b);
});
test('seven-day window expires old reviews and includes the cutoff review', () => {
  const results = features(normalize([row('a', 'u1', '01'), row('b', 'u1', '15')]), cutoff);
  assert.equal(results.length, 2);
  assert.equal(results[1].user_prior_7d_count, 0);
});
test('EDA fills missing days, uses Monday weeks, identifies planted spike', () => {
  const rows = [row('base', 'u1', '01'), ...Array.from({ length: 5 }, (_, i) => row(`s${i}`, `u${i}`, '10'))];
  const result = eda(normalize(rows), cutoff);
  assert.equal(result.daily.length, 15);
  assert.equal(result.daily[1].review_count, 0);
  assert.equal(result.weekly[0].week_start, '2024-01-01');
  assert.equal(result.weekly.reduce((s, r) => s + r.review_count, 0), rows.length);
  const spike = result.productDaily.find(r => r.date === '2024-01-10');
  assert.equal(spike.prior_7d_daily_mean, 0);
  assert.equal(spike.spike_candidate, 1);
  assert.throws(() => eda([], cutoff), /No reviews/);
});
