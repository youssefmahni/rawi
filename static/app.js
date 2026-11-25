// Rawi page: mic → transcribe → streamed answer → spoken reply in the answer's dialect.
import { SentenceBuffer, cleanForSpeech } from './speech.js';
import { DIALECTS, languageOf } from './dialects.js';

const $ = (id) => document.getElementById(id);
const messages = $('messages');
const form = $('ask-form');
const input = $('question');
const mic = $('mic');

const MIC_DENIED = "Microphone access was denied. Allow it in the browser's site settings.";
const DIDNT_CATCH = "Didn't catch that";
const MIN_RECORDING_MS = 500;
const MAX_RECORDING_MS = 60000; // auto-stop and send after a minute
const MIC_LABELS = {
  idle: 'Ask by voice',
  listening: 'Stop recording and send',
  transcribing: 'Transcribing',
  thinking: 'Stop answer',
  speaking: 'Stop speaking',
};

// Playback speed of the spoken reply (1 = normal; 1.2 = 20% faster, natural pitch kept).
const SPEECH_RATE = 1.4;

// ---- state ----
const state = {
  phase: 'idle', // idle → listening → transcribing → thinking → speaking → idle
  history: [], // last 6 messages: {role, content}
  controller: null, // AbortController of the running /api/ask
  sentAt: 0,
  firstAudioLogged: false,
  dialect: null, // dialect of the current turn, from the dialect SSE event
  pendingSpeech: [], // sentences that arrived before the dialect event (defensive)
};

function setPhase(phase) {
  state.phase = phase;
  mic.dataset.state = phase;
  mic.setAttribute('aria-label', MIC_LABELS[phase]);
  mic.setAttribute('aria-disabled', String(phase === 'transcribing'));
}

function settleIfDone() {
  if (!state.controller && !audioQ.busy && !audioQ.items.length && state.phase !== 'listening'
      && state.phase !== 'transcribing') setPhase('idle');
}

// ---- messages ----
function nearBottom() {
  return messages.scrollTop + messages.clientHeight >= messages.scrollHeight - 80; // the window scrolls, not the page
}

function append(node) {
  const stick = nearBottom();
  $('empty-hint')?.remove();
  messages.append(node);
  if (stick) messages.scrollTop = messages.scrollHeight;
  return node;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const systemMessage = (text) => append(el('p', 'system-message', text));
const errorMessage = (text) => append(el('p', 'system-message error', text)); // primary label color for legibility

// ---- mic ----
const recording = { stream: null, recorder: null, chunks: [], ctx: null, raf: 0, session: 0 };

async function startListening() {
  const session = ++recording.session;
  setPhase('listening');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (session !== recording.session) return;
    setPhase('idle');
    errorMessage(err.name === 'NotAllowedError' ? MIC_DENIED
      : err.name === 'NotFoundError' ? 'No microphone found.' : 'Microphone unavailable.');
    return;
  }
  if (session !== recording.session || state.phase !== 'listening') {
    stream.getTracks().forEach((t) => t.stop()); // released before the mic opened
    return;
  }
  recording.stream = stream;
  recording.chunks = [];
  recording.recorder = new MediaRecorder(stream);
  recording.recorder.ondataavailable = (e) => { if (e.data.size) recording.chunks.push(e.data); };
  recording.recorder.onstop = sendRecording;
  recording.recorder.start();
  recording.startedAt = performance.now();
  recording.timer = setTimeout(() => {
    if (session === recording.session) stopListening();
  }, MAX_RECORDING_MS);
  startLevelMeter(stream);
}

function startLevelMeter(stream) {
  recording.ctx ??= new AudioContext();
  recording.ctx.resume();
  const source = recording.ctx.createMediaStreamSource(stream);
  const analyser = recording.ctx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  const tick = () => {
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((sum, v) => sum + v * v, 0) / samples.length);
    mic.style.setProperty('--level', Math.min(1, rms * 6).toFixed(3));
    recording.raf = requestAnimationFrame(tick);
  };
  tick();
  recording.stopMeter = () => {
    cancelAnimationFrame(recording.raf);
    source.disconnect();
    mic.style.setProperty('--level', '0');
  };
}

function stopListening() {
  if (state.phase !== 'listening') return; // a repeat stop (double tap, click + Space keyup) is a no-op
  clearTimeout(recording.timer);
  state.sentAt = performance.now();
  if (recording.recorder?.state === 'recording') {
    setPhase('transcribing'); // before stop(): onstop fires later, and the phase must leave 'listening' now
    recording.recorder.stop(); // → sendRecording
  } else {
    recording.session++; // mic never opened (very short tap)
    setPhase('idle');
    systemMessage(DIDNT_CATCH);
  }
}

async function sendRecording() {
  const { recorder, stream } = recording;
  recording.stopMeter?.();
  stream.getTracks().forEach((t) => t.stop());
  recording.recorder = recording.stream = null;
  const blob = new Blob(recording.chunks, { type: recorder.mimeType });
  // A tap shorter than MIN_RECORDING_MS yields a clip too short to decode; no question is that short.
  if (!blob.size || state.sentAt - recording.startedAt < MIN_RECORDING_MS) {
    setPhase('idle');
    systemMessage(DIDNT_CATCH);
    return;
  }
  setPhase('transcribing');
  try {
    const r = await fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'application/octet-stream' },
      body: blob,
    });
    if (!r.ok) throw new Error(await errorDetail(r, 'Transcription failed.'));
    const { text, language } = await r.json();
    if (!text) {
      setPhase('idle');
      systemMessage(DIDNT_CATCH);
      return;
    }
    ask(text, state.sentAt, language);
  } catch (err) {
    setPhase('idle');
    errorMessage(err.message || 'Transcription failed.');
  }
}

async function errorDetail(response, fallback) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : fallback;
  } catch {
    return fallback;
  }
}

// ---- ask / SSE ----
async function* readSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let end;
    while ((end = buf.indexOf('\n\n')) !== -1) {
      const block = buf.slice(0, end);
      buf = buf.slice(end + 2);
      let event = 'message';
      const data = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield { event, data: JSON.parse(data.join('\n')) };
    }
  }
}

async function ask(question, sentAt = performance.now(), language = languageOf(question)) {
  stopAnswer();
  const userBubble = el('div', 'bubble user', question);
  userBubble.dir = 'auto';
  append(userBubble);
  const turn = append(el('div', 'turn'));
  const bubble = el('div', 'bubble assistant');
  bubble.setAttribute('aria-busy', 'true');
  turn.append(bubble);
  bubble.hidden = true;

  const controller = new AbortController();
  state.controller = controller;
  state.sentAt = sentAt;
  state.firstAudioLogged = false;
  state.dialect = null;
  state.pendingSpeech = [];
  setPhase('thinking');
  const buffer = new SentenceBuffer();
  let answer = '';
  let badge = null;

  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: state.history, language }),
      signal: controller.signal,
    });
    if (!r.ok) throw new Error(await errorDetail(r, 'The request failed. Try again.'));
    for await (const { event, data } of readSSE(r.body)) {
      if (event === 'dialect') {
        const d = DIALECTS[data.dialect] || DIALECTS.en;
        state.dialect = data.dialect;
        bubble.dir = d.dir;
        bubble.lang = d.lang;
        badge = el('span', 'dialect-badge', `${d.native} · ${d.label}`);
        turn.prepend(badge);
        state.pendingSpeech.splice(0).forEach(speak);
      } else if (event === 'delta') {
        const stick = nearBottom();
        answer += data.text;
        bubble.hidden = false;
        bubble.append(data.text);
        if (stick) messages.scrollTop = messages.scrollHeight;
        buffer.push(data.text).forEach(speak);
      } else if (event === 'done') {
        buffer.flush().forEach(speak);
        state.history = [...state.history,
          { role: 'user', content: question }, { role: 'assistant', content: answer }].slice(-6);
      } else if (event === 'error') {
        errorMessage(data.message);
        if (data.kind === 'offline') speak(data.message, data.language === 'ar' ? 'ar-SA' : 'en');
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') errorMessage(err.message || 'The request failed. Try again.');
  } finally {
    bubble.setAttribute('aria-busy', 'false');
    if (!answer) {
      bubble.remove();
      badge?.remove();
    }
    if (!turn.childElementCount) turn.remove();
    if (state.controller === controller) {
      state.controller = null;
      settleIfDone();
    }
  }
}

function stopAnswer() {
  state.controller?.abort();
  state.controller = null;
  clearAudio();
  setPhase('idle');
}

// ---- audio queue ----
const audioQ = {
  items: [], busy: false, gen: 0, abort: new AbortController(), el: new Audio(), finish: null,
  failedGen: -1, // the answer (gen) whose speech failure is already on screen: one message per answer
};

function speechFailed(gen, message) {
  console.warn('Skipping sentence:', message);
  if (gen !== audioQ.gen || audioQ.failedGen === gen) return;
  audioQ.failedGen = gen;
  errorMessage(message);
}

function speak(sentence, dialectOverride) {
  const dialect = dialectOverride || state.dialect;
  if (!dialect) {
    state.pendingSpeech.push(sentence); // sentence before the dialect event (server sends dialect first)
    return;
  }
  const text = cleanForSpeech(sentence).slice(0, 1000);
  if (!/[\p{L}\p{N}]/u.test(text)) return; // Piper fails on punctuation-only text
  audioQ.items.push({ text, dialect, blob: null });
  if (audioQ.busy) prefetch(audioQ.items[0]); // the next sentence loads while one plays
  pump();
}

function prefetch(item) {
  if (!item || item.blob) return;
  const { gen } = audioQ;
  item.blob = fetch('/api/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: item.text, dialect: item.dialect }),
    signal: audioQ.abort.signal,
  })
    .then(async (r) => {
      if (r.ok) return r.blob();
      speechFailed(gen, await errorDetail(r, 'Speech synthesis failed.'));
      return null;
    })
    .catch((err) => {
      if (err.name !== 'AbortError') speechFailed(gen, 'Speech request failed.');
      return null;
    });
}

async function pump() {
  if (audioQ.busy) return;
  audioQ.busy = true;
  const gen = audioQ.gen;
  while (gen === audioQ.gen && audioQ.items.length) {
    const item = audioQ.items.shift();
    prefetch(item);
    prefetch(audioQ.items[0]);
    const blob = await item.blob;
    if (blob && gen === audioQ.gen) await play(blob);
  }
  if (gen !== audioQ.gen) return; // cleared; clearAudio reset the queue
  audioQ.busy = false;
  settleIfDone();
}

function play(blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = audioQ.el;
    let done = false;
    const finish = () => { // idempotent: ended, error, a rejected play() and clearAudio may all call it
      if (done) return;
      done = true;
      if (audioQ.finish === finish) {
        audioQ.finish = null;
        audio.onended = audio.onerror = null;
      }
      URL.revokeObjectURL(url);
      resolve();
    };
    audioQ.finish = finish;
    audio.onended = finish;
    audio.onerror = finish;
    audio.preservesPitch = true;
    audio.playbackRate = SPEECH_RATE;
    audio.src = url;
    audio.play().then(() => {
      if (!state.firstAudioLogged) {
        state.firstAudioLogged = true;
        console.info('first audio after', Math.round(performance.now() - state.sentAt));
      }
      if (state.phase === 'thinking') setPhase('speaking');
    }, finish);
  });
}

function clearAudio() {
  audioQ.gen++;
  audioQ.abort.abort();
  audioQ.abort = new AbortController();
  audioQ.items = [];
  audioQ.busy = false;
  audioQ.el.pause();
  audioQ.finish?.();
}

// ---- health ----
async function checkHealth() {
  const status = $('status');
  const label = $('status-label');
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    const missing = Object.entries(h.voices)
      .filter(([, ok]) => !ok)
      .map(([d]) => DIALECTS[d]?.label)
      .filter(Boolean);
    if (!h.deepseek_key) [status.dataset.health, label.textContent] = ['nokey', 'No DeepSeek key'];
    else if (missing.length) [status.dataset.health, label.textContent] = ['down', `Voice missing: ${missing.join(', ')}`];
    else [status.dataset.health, label.textContent] = ['ok', 'Ready'];
  } catch {
    [status.dataset.health, label.textContent] = ['down', 'Server unreachable'];
  }
}

// ---- input and keyboard ----
mic.addEventListener('click', () => {
  if (state.phase === 'idle') startListening();
  else if (state.phase === 'listening') stopListening();
  else if (state.phase === 'thinking' || state.phase === 'speaking') stopAnswer();
});

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || state.phase === 'listening' || state.phase === 'transcribing') return;
  input.value = '';
  ask(question);
});

// Space is push-to-talk unless another control would use it.
const spaceIgnored = (target) => target.closest('input, select, textarea, summary, a, button:not(#mic)');

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && (state.phase === 'thinking' || state.phase === 'speaking')) {
    stopAnswer();
  } else if (e.code === 'Space' && !spaceIgnored(e.target)) {
    e.preventDefault();
    if (!e.repeat && state.phase === 'idle') {
      recording.ptt = true;
      startListening();
    }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code !== 'Space' || spaceIgnored(e.target)) return;
  e.preventDefault();
  if (recording.ptt && state.phase === 'listening') stopListening();
  recording.ptt = false;
});

// ---- init ----
setPhase('idle');
checkHealth();
setInterval(checkHealth, 15000);
