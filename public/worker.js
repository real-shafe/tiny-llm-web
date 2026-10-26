// Loads the model and runs generation off the main thread so the page never janks.
import { TinyGPT } from './model.js'
import { Tiktoken, gpt2 } from './vendor/tiktoken-gpt2.js'

// When the 128-token window fills, keep this many recent tokens and rebuild the
// cache from them. Position embeddings are absolute, so the cache can't just slide.
const KEEP = 96

let model = null
let enc = null
let vocabIndex = null
let running = null

async function fetchWithProgress(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body.getReader()
  const parts = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
    loaded += value.length
    postMessage({ type: 'progress', loaded, total })
  }
  const out = new Uint8Array(loaded)
  let o = 0
  for (const p of parts) { out.set(p, o); o += p.length }
  return out.buffer
}

async function load() {
  try {
    const manifest = await (await fetch('model/manifest.json')).json()
    const buffer = await fetchWithProgress('model/weights.bin')
    model = new TinyGPT(manifest, buffer)
    enc = new Tiktoken(gpt2)
    vocabIndex = new Map(manifest.vocab.map((id, i) => [id, i]))
    postMessage({
      type: 'ready',
      info: {
        paramsTrained: manifest.params_trained,
        paramsExported: manifest.params_exported,
        vocab: manifest.vocab.length,
        valLoss: manifest.val_loss,
        bytes: buffer.byteLength,
      },
    })
  } catch (err) {
    postMessage({ type: 'error', message: String(err.message || err) })
  }
}

function softmax(logits, temperature) {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const p = new Float32Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) sum += (p[i] = Math.exp((logits[i] - max) / temperature))
  for (let i = 0; i < p.length; i++) p[i] /= sum
  return p
}

// Sample from probs, restricted to the topK most likely tokens (0 = no limit).
function sample(probs, topK) {
  let floor = 0
  if (topK > 0 && topK < probs.length) floor = probs.slice().sort()[probs.length - topK]
  let total = 0
  for (let i = 0; i < probs.length; i++) if (probs[i] >= floor) total += probs[i]
  let r = Math.random() * total
  let last = 0
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] < floor) continue
    last = i
    r -= probs[i]
    if (r <= 0) return i
  }
  return last
}

// TinyStories has curly quotes double-encoded as Windows-1252 (“ became â€œ), and
// the closing quote lost its last byte entirely, so the model writes them that
// way. Show the characters they were meant to be.
const QUOTES = { 'â€œ': '“', 'â€˜': '‘', 'â€™': '’', 'â€“': '–', 'â€”': '—', 'â€¦': '…' }
const fixQuotes = (s) => s.replace(/â€(.?)/gs, (m, c) => QUOTES[m] ?? (c ? `”${c}` : m))

// Stories have blank lines between their own paragraphs, so a blank line alone
// doesn't end one. Most stories open with one of these, so a paragraph that
// starts with one means the model has moved on to a new story.
const OPENERS = ['Once upon a time', 'Once there was', 'Once, there was']

// Returns { end } if a new story has begun at that index, otherwise { safe }: how
// much of gen can be shown without risking showing the start of a new story or
// half a broken quote.
function settle(gen, stopAtEnd) {
  let safe = gen.length
  const partial = gen.match(/â€?$/)
  if (partial) safe = partial.index
  if (!stopAtEnd) return { safe }
  for (const b of gen.matchAll(/\n\s*\n\s*/g)) {
    const after = gen.slice(b.index + b[0].length)
    if (OPENERS.some((o) => after.startsWith(o))) return { end: b.index }
    if (OPENERS.some((o) => o.startsWith(after))) return { safe: Math.min(safe, b.index) }
  }
  // A trailing newline may be the first half of a blank line.
  const nl = gen.search(/\n\s*$/)
  return { safe: nl === -1 ? safe : Math.min(safe, nl) }
}

// Yield between tokens so a stop message can get in. A MessageChannel hop isn't
// clamped to 4ms the way nested setTimeout is.
const tick = new MessageChannel()
const waiting = []
tick.port1.onmessage = () => waiting.shift()()
const nextTick = () => new Promise((r) => { waiting.push(r); tick.port2.postMessage(0) })

async function generate({ id, prompt, temperature, topK, maxTokens, stopAtEnd }) {
  const job = { id, stopped: false }
  job.finished = new Promise((r) => { job.finish = r })
  running = job

  const ids = enc.encode(prompt)
  const pieces = ids.map((t) => ({ text: enc.decode([t]), known: vocabIndex.has(t) }))
  postMessage({ type: 'prompt', id, pieces })
  const known = ids.filter((t) => vocabIndex.has(t)).map((t) => vocabIndex.get(t))
  if (!known.length) {
    postMessage({ type: 'done', id, count: 0, ms: 0, reason: 'unknown' })
    running = null
    job.finish()
    return
  }

  const ctx = []
  let logits = null
  const feed = (t) => {
    if (model.pos === model.blockSize) {
      model.reset()
      for (const u of ctx.slice(-KEEP)) model.step(u)
    }
    logits = model.step(t)
    ctx.push(t)
  }

  model.reset()
  for (const t of known) feed(t)

  const t0 = performance.now()
  let count = 0
  let pending = []
  let rawText = ''
  let sent = 0
  let lastP = 1
  let reason = 'length'
  const emit = (text) => {
    if (text) postMessage({ type: 'token', id, text, p: lastP })
    sent += text.length
  }
  while (count < maxTokens) {
    if (job.stopped) { reason = 'stopped'; break }
    const raw = softmax(logits, 1)
    const probs = temperature === 1 ? raw : softmax(logits, temperature)
    const next = sample(probs, topK)
    count++
    lastP = raw[next]
    pending.push(model.vocab[next])

    // A token can end partway through a multi-byte character; hold it back
    // until the character is complete.
    const chunk = enc.decode(pending)
    if (!chunk.endsWith('�')) {
      pending = []
      rawText += chunk
      const gen = fixQuotes(rawText)
      const s = settle(gen, stopAtEnd)
      if (s.end !== undefined) {
        emit(gen.slice(sent, s.end))
        reason = 'end'
        break
      }
      if (s.safe > sent) emit(gen.slice(sent, s.safe))
    }

    feed(next)
    await nextTick()
  }
  if (reason !== 'end') emit(fixQuotes(rawText).slice(sent))
  postMessage({ type: 'done', id, count, ms: performance.now() - t0, reason })
  if (running === job) running = null
  job.finish()
}

onmessage = async (e) => {
  const msg = e.data
  if (msg.type === 'generate') {
    // The model's cache is shared, so the previous run has to finish first.
    const prev = running
    if (prev) { prev.stopped = true; await prev.finished }
    generate(msg)
  } else if (msg.type === 'stop' && running) {
    running.stopped = true
  }
}

load()
