const $ = (id) => document.getElementById(id)
const els = {
  prompt: $('prompt'), go: $('go'), stop: $('stop'), status: $('status'), bar: $('bar'),
  story: $('story'), temp: $('temp'), len: $('len'), topk: $('topk'),
  stopAtEnd: $('stopAtEnd'), showTokens: $('showTokens'),
  unknownNote: $('unknownNote'), tokenNote: $('tokenNote'),
}

const fmt = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} million` : n.toLocaleString())

for (const [input, out, show] of [
  [els.temp, $('tempOut'), (v) => Number(v).toFixed(2)],
  [els.len, $('lenOut'), (v) => v],
  [els.topk, $('topkOut'), (v) => (Number(v) === 0 ? 'off' : v)],
]) {
  const sync = () => { out.textContent = show(input.value) }
  input.addEventListener('input', sync)
  sync()
}

$('chips').addEventListener('click', (e) => {
  const chip = e.target.closest('.chip')
  if (!chip) return
  els.prompt.value = chip.textContent
  els.prompt.focus()
})

els.showTokens.addEventListener('change', () => {
  els.story.classList.toggle('showTokens', els.showTokens.checked)
  els.tokenNote.hidden = !els.showTokens.checked
})

const worker = new Worker('worker.js', { type: 'module' })
let ready = false
let jobId = 0
let generating = false

function setStatus(text, err = false) {
  els.status.textContent = text
  els.status.classList.toggle('err', err)
}

function setGenerating(on) {
  generating = on
  els.go.textContent = on ? 'Writing…' : 'Generate'
  els.go.disabled = on || !ready
  els.stop.hidden = !on
  els.story.classList.toggle('cursor', on)
}

function start() {
  if (!ready || !els.prompt.value.trim()) return
  jobId++
  els.story.textContent = ''
  els.unknownNote.hidden = true
  setGenerating(true)
  setStatus('')
  worker.postMessage({
    type: 'generate',
    id: jobId,
    prompt: els.prompt.value,
    temperature: Number(els.temp.value),
    topK: Number(els.topk.value),
    maxTokens: Number(els.len.value),
    stopAtEnd: els.stopAtEnd.checked,
  })
}

els.go.addEventListener('click', start)
els.stop.addEventListener('click', () => worker.postMessage({ type: 'stop' }))
els.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); start() }
})

worker.onmessage = (e) => {
  const m = e.data
  if (m.type === 'progress') {
    if (m.total) {
      els.bar.firstElementChild.style.width = `${(100 * m.loaded) / m.total}%`
      setStatus(`Downloading the model… ${(m.loaded / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB`)
    } else {
      setStatus(`Downloading the model… ${(m.loaded / 1e6).toFixed(1)} MB`)
    }
  } else if (m.type === 'ready') {
    ready = true
    els.bar.hidden = true
    const { paramsTrained, paramsExported, vocab, valLoss, bytes } = m.info
    $('paramCount').textContent = fmt(paramsExported)
    $('aboutNumbers').textContent =
      `It uses GPT-2's tokenizer, but only ${vocab.toLocaleString()} of its 50,257 tokens ever showed up in training. ` +
      `The model has ${fmt(paramsTrained)} parameters as trained; dropping the rows for tokens it never saw leaves ` +
      `${fmt(paramsExported)}, a ${(bytes / 1e6).toFixed(1)} MB download. Its validation loss is ${valLoss.toFixed(2)}.`
    setGenerating(false)
    setStatus(`Ready. Model loaded: ${(bytes / 1e6).toFixed(1)} MB.`)
  } else if (m.type === 'error') {
    setStatus(`Couldn't load the model: ${m.message}`, true)
    els.go.textContent = 'Unavailable'
  } else if (m.id !== jobId) {
    // A message from a run that was replaced.
  } else if (m.type === 'prompt') {
    const span = document.createElement('span')
    span.className = 'prompt'
    let unknown = false
    for (const p of m.pieces) {
      if (p.known) span.append(p.text)
      else {
        unknown = true
        const s = document.createElement('span')
        s.className = 'unknown'
        s.title = 'Never seen in training, so it was skipped'
        s.textContent = p.text
        span.append(s)
      }
    }
    els.story.append(span)
    els.unknownNote.hidden = !unknown
  } else if (m.type === 'token') {
    const s = document.createElement('span')
    s.className = m.p < 0.02 ? 'tok unlikely' : 'tok'
    s.title = `${(m.p * 100).toFixed(m.p < 0.01 ? 2 : 1)}% likely`
    s.textContent = m.text
    els.story.append(s)
  } else if (m.type === 'done') {
    setGenerating(false)
    if (m.reason === 'unknown') {
      setStatus("None of that prompt appeared in the training data. Try simpler words.", true)
      return
    }
    const rate = m.ms > 0 ? Math.round((m.count * 1000) / m.ms) : 0
    const why = { end: 'The story ended.', stopped: 'Stopped.', length: 'Hit the length limit.' }[m.reason]
    setStatus(`${why} ${m.count} tokens at ${rate} tokens/s.`)
  }
}
