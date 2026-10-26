// The tiny_llm GPT forward pass in plain JS, mirroring src/utils/model.py:
// pre-norm blocks, multi-head causal attention, a 4x GELU MLP, learned position
// embeddings. Tokens are fed one at a time with a key/value cache, so each new
// token only does one position's worth of work instead of re-running the window.

function fp16ToFloat32(u16) {
  const out = new Float32Array(u16.length)
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i]
    const s = h & 0x8000 ? -1 : 1
    const e = (h >> 10) & 0x1f
    const m = h & 0x3ff
    if (e === 0) out[i] = s * m * 2 ** -24
    else if (e === 31) out[i] = m ? NaN : s * Infinity
    else out[i] = s * (1 + m / 1024) * 2 ** (e - 15)
  }
  return out
}

// out[r] = W[r, :] . x + b[r], with W stored row-major as (rows, cols) like nn.Linear.
function matvec(out, W, x, b, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let sum = b ? b[r] : 0
    const o = r * cols
    for (let c = 0; c < cols; c++) sum += W[o + c] * x[c]
    out[r] = sum
  }
}

function layerNorm(out, x, w, b, n) {
  let mean = 0
  for (let i = 0; i < n; i++) mean += x[i]
  mean /= n
  let v = 0
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2
  const inv = 1 / Math.sqrt(v / n + 1e-5)
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i]
}

// Abramowitz & Stegun 7.1.26, max error 1.5e-7. PyTorch's F.gelu default is the
// exact erf form, not the tanh approximation.
function erf(x) {
  const s = x < 0 ? -1 : 1
  x = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * x)
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return s * y
}

const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2))

export class TinyGPT {
  constructor(manifest, buffer) {
    const { n_embd, n_head, n_layer, block_size } = manifest.config
    Object.assign(this, { C: n_embd, nHead: n_head, nLayer: n_layer, blockSize: block_size })
    this.vocab = manifest.vocab
    this.V = manifest.vocab.length
    this.paramsTrained = manifest.params_trained
    this.paramsExported = manifest.params_exported

    const all = fp16ToFloat32(new Uint16Array(buffer))
    this.w = {}
    for (const t of manifest.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1)
      this.w[t.name] = all.subarray(t.offset, t.offset + size)
    }

    const C = this.C
    this.kCache = Array.from({ length: n_layer }, () => new Float32Array(block_size * C))
    this.vCache = Array.from({ length: n_layer }, () => new Float32Array(block_size * C))
    this.x = new Float32Array(C)
    this.h = new Float32Array(C)
    this.q = new Float32Array(C)
    this.att = new Float32Array(C)
    this.tmp = new Float32Array(C)
    this.ff = new Float32Array(4 * C)
    this.scores = new Float32Array(block_size)
    this.logits = new Float32Array(this.V)
    this.pos = 0
  }

  reset() {
    this.pos = 0
  }

  // Feed one token (an index into this.vocab) at the next position. Returns the
  // logits for the token after it. Callers must reset() before pos hits blockSize.
  step(token) {
    const { C, nHead, w, x, h, q, att, tmp, ff, scores } = this
    const hs = C / nHead
    const p = this.pos
    const scale = 1 / Math.sqrt(hs)

    const te = w.tok_emb.subarray(token * C, (token + 1) * C)
    const pe = w.pos_emb.subarray(p * C, (p + 1) * C)
    for (let i = 0; i < C; i++) x[i] = te[i] + pe[i]

    for (let l = 0; l < this.nLayer; l++) {
      const K = this.kCache[l]
      const Vc = this.vCache[l]
      layerNorm(h, x, w[`${l}.ln1_w`], w[`${l}.ln1_b`], C)
      matvec(q, w[`${l}.wq`], h, null, C, C)
      matvec(K.subarray(p * C, (p + 1) * C), w[`${l}.wk`], h, null, C, C)
      matvec(Vc.subarray(p * C, (p + 1) * C), w[`${l}.wv`], h, null, C, C)

      for (let hd = 0; hd < nHead; hd++) {
        const o = hd * hs
        let max = -Infinity
        for (let t = 0; t <= p; t++) {
          let s = 0
          for (let i = 0; i < hs; i++) s += q[o + i] * K[t * C + o + i]
          s *= scale
          scores[t] = s
          if (s > max) max = s
        }
        let sum = 0
        for (let t = 0; t <= p; t++) sum += (scores[t] = Math.exp(scores[t] - max))
        for (let i = 0; i < hs; i++) att[o + i] = 0
        for (let t = 0; t <= p; t++) {
          const a = scores[t] / sum
          for (let i = 0; i < hs; i++) att[o + i] += a * Vc[t * C + o + i]
        }
      }
      matvec(tmp, w[`${l}.wo`], att, w[`${l}.bo`], C, C)
      for (let i = 0; i < C; i++) x[i] += tmp[i]

      layerNorm(h, x, w[`${l}.ln2_w`], w[`${l}.ln2_b`], C)
      matvec(ff, w[`${l}.w1`], h, w[`${l}.b1`], 4 * C, C)
      for (let i = 0; i < 4 * C; i++) ff[i] = gelu(ff[i])
      matvec(tmp, w[`${l}.w2`], ff, w[`${l}.b2`], C, 4 * C)
      for (let i = 0; i < C; i++) x[i] += tmp[i]
    }

    layerNorm(h, x, w.lnf_w, w.lnf_b, C)
    matvec(this.logits, w.lm_w, h, w.lm_b, this.V, C)
    this.pos++
    return this.logits
  }
}
