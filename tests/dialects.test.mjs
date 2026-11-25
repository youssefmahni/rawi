import test from 'node:test';
import assert from 'node:assert/strict';
import { DIALECTS, hasArabic, languageOf } from '../static/dialects.js';

test('four dialects with direction', () => {
  assert.deepEqual(Object.keys(DIALECTS), ['en', 'ar-MA', 'ar-EG', 'ar-SA']);
  assert.equal(DIALECTS['ar-MA'].dir, 'rtl');
  assert.equal(DIALECTS.en.dir, 'ltr');
});

test('language of typed text', () => {
  assert.equal(languageOf('عندي meeting'), 'ar');
  assert.equal(languageOf('hello'), 'en');
  assert.ok(hasArabic('هلا') && !hasArabic('3.5'));
});
