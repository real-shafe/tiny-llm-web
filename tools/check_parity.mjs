// Checks the JS forward pass against PyTorch. Run after export_weights.py:
//   node tools/check_parity.mjs
import { readFileSync } from 'node:fs'
import { TinyGPT } from '../public/model.js'

const root = new URL('..', import.meta.url)
const manifest = JSON.parse(readFileSync(new URL('public/model/manifest.json', root)))
const buf = readFileSync(new URL('public/model/weights.bin', root))
const fixture = JSON.parse(readFileSync(new URL('tools/fixture.json', root)))

const model = new TinyGPT(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const index = new Map(manifest.vocab.map((id, i) => [id, i]))

let logits
for (const id of fixture.ids) logits = model.step(index.get(id))

let maxDiff = 0
for (let i = 0; i < logits.length; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - fixture.logits[i]))
const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0)
const sameTop = argmax(Array.from(logits)) === argmax(fixture.logits)

// Rough speed: fill the whole context window one token at a time.
model.reset()
const t0 = performance.now()
for (let p = 0; p < model.blockSize; p++) model.step(p % model.V)
const ms = performance.now() - t0

console.log(`prompt        : ${JSON.stringify(fixture.prompt)}`)
console.log(`max |Δlogit|  : ${maxDiff.toFixed(4)} (fp16 weights vs fp32 PyTorch)`)
console.log(`same top token: ${sameTop}`)
console.log(`speed         : ${Math.round((model.blockSize * 1000) / ms)} tokens/s over a full window`)
if (!sameTop || maxDiff > 0.1) process.exit(1)
