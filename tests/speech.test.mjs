import test from 'node:test';
import assert from 'node:assert/strict';
import { SentenceBuffer, cleanForSpeech } from '../static/speech.js';

test('emits sentences as they complete', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('ECC 2-2-3 requires MFA. It app'), ['ECC 2-2-3 requires MFA.']);
  assert.deepEqual(b.push('lies to admins! Done'), ['It applies to admins!']);
  assert.deepEqual(b.flush(), ['Done']);
});

test('does not split decimals or abbreviations', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('Version 1.0 of the policy, e.g. the access policy, applies. Next'),
    ['Version 1.0 of the policy, e.g. the access policy, applies.']);
});

test('flush of whitespace is empty', () => {
  const b = new SentenceBuffer();
  b.push('One. ');
  assert.deepEqual(b.flush(), []);
});

test('cleanForSpeech strips markdown', () => {
  assert.equal(cleanForSpeech('**MFA** is `required`\n# now'), 'MFA is required now');
});

test('numbered list markers do not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('Steps:\n1. Enable MFA\n2. Review logs. Done'),
    ['Steps:\n1. Enable MFA\n2. Review logs.']);
  assert.deepEqual(b.flush(), ['Done']);
});

test('abbreviation at true sentence end splits before uppercase', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('The answer is no. Next point'), ['The answer is no.']);
});

test('No. followed by a digit does not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('See No. 5 for details. X'), ['See No. 5 for details.']);
});

test('abbreviation split across chunks waits for next character', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('Use e.g'), []);
  assert.deepEqual(b.push('. the policy. Next'), ['Use e.g. the policy.']);
});

test('splits on Arabic question mark', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('وش أخبارك؟ أنا '), ['وش أخبارك؟']);
  assert.deepEqual(b.flush(), ['أنا']);
});

test('Arabic comma does not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('هلا والله، كيف حالك؟ '), ['هلا والله، كيف حالك؟']);
});

test('decimal number does not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('It costs 3.5 riyals. Ok'), ['It costs 3.5 riyals.']);
});

test('period before Arabic splits', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('تمام. شكراً '), ['تمام.']);
});
