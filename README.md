# tiny_llm in the browser

A small GPT trained from scratch on TinyStories, running entirely client-side. The model and training code live in [real-shafe/tiny_llm](https://github.com/real-shafe/tiny_llm); this repo is the page that runs it.

**Try it:** https://tiny-llm.shafe.dev

- The forward pass is plain JavaScript in a Web Worker (`public/model.js`), with a key/value cache
- GPT-2 tokenizer via [js-tiktoken](https://github.com/dqbd/tiktoken) (MIT), bundled into `public/vendor/tiktoken-gpt2.js`
- Weights ship as fp16. Only the ~8.5K tokens that appear in the training data are kept, which shrinks the download about 5x

## Updating the model

With [tiny_llm](https://github.com/real-shafe/tiny_llm) checked out next to this repo and trained (`checkpoints/gpt.pt` exists):

```bash
uv run --project ../tiny_llm python tools/export_weights.py
node tools/check_parity.mjs
```

The export writes `public/model/` and a reference fixture. The parity check runs the JS model against the PyTorch logits.

## Run locally

```bash
python3 -m http.server -d public 8000
```

## Deploy

Served by a Cloudflare Worker with static assets (`wrangler.jsonc`). Deploy with:

```bash
npx wrangler deploy
```
