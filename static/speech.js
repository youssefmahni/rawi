// Sentence splitting and markdown cleanup for streaming text-to-speech.

const ABBREVIATIONS = /(?:^|[^a-z])(?:e\.g|i\.e|etc|vs|no|mr|ms|dr)\.$/i;

// A "1." / "12." that stands alone at the start of the text or of a line.
const LIST_NUMBER = /(?:^|\n)\s*\d{1,2}\.$/;

export function cleanForSpeech(text) {
  return text.replace(/[*_`#]/g, '').replace(/\s+/g, ' ').trim();
}

export class SentenceBuffer {
  constructor() {
    this.buf = '';
  }

  push(delta) {
    this.buf += delta;
    const out = [];
    const re = /[.!?؟](\s+)(\S)?/g;
    let start = 0;
    let m;
    while ((m = re.exec(this.buf)) !== null) {
      const end = m.index + 1;
      const candidate = this.buf.slice(start, end);
      if (m[0][0] === '.') {
        if (LIST_NUMBER.test(candidate)) continue;
        if (ABBREVIATIONS.test(candidate)) {
          // Wait for the next character; split only before an uppercase Latin letter,
          // or before a non-Latin letter (Arabic has no case, so it still ends the sentence).
          if (m[2] === undefined) break;
          const next = m[2];
          const split = /[A-Z]/.test(next) || (/\p{L}/u.test(next) && !/[A-Za-z]/.test(next));
          if (!split) continue;
        }
      }
      const sentence = candidate.trim();
      if (sentence) out.push(sentence);
      start = end;
    }
    this.buf = this.buf.slice(start);
    return out;
  }

  flush() {
    const rest = this.buf.trim();
    this.buf = '';
    return rest ? [rest] : [];
  }
}
