"""Export tiny_llm's checkpoints/gpt.pt into the files the browser loads.

Run from this repo's root, using tiny_llm's environment:

    uv run --project ../tiny_llm python tools/export_weights.py

Writes public/model/manifest.json + public/model/weights.bin (fp16), and
tools/fixture.json for check_parity.mjs.

The training data only ever uses ~8.5K of GPT-2's 50,257 tokens. The other rows
of the embedding and lm_head were never a training target, so the model gives
them almost no probability. Keeping only the seen tokens cuts the download by ~5x;
the script prints how much probability mass that throws away so you can check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TINY_LLM = ROOT.parent / "tiny_llm"
sys.path.insert(0, str(TINY_LLM / "src"))

from utils.model import GPT  # noqa: E402

CKPT_PATH = TINY_LLM / "checkpoints" / "gpt.pt"
TOKENS_PATH = TINY_LLM / "data" / "tinystories_tokens.npy"
OUT_DIR = ROOT / "public" / "model"
FIXTURE_PATH = ROOT / "tools" / "fixture.json"
FIXTURE_PROMPT = "Once upon a time, there was a little dog named Max. He liked to"


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    model = GPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = {k: v.float() for k, v in model.state_dict().items()}

    kept = np.unique(np.load(TOKENS_PATH)).astype(np.int64)
    kept_t = torch.from_numpy(kept)
    C, n_head, n_layer = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]

    tensors: list[tuple[str, torch.Tensor]] = [
        ("tok_emb", sd["token_embedding.weight"][kept_t]),
        ("pos_emb", sd["position_embedding.weight"]),
    ]
    for l in range(n_layer):
        p = f"blocks.{l}."
        # nn.Linear stores (out, in). Stacking the heads' rows gives the same
        # layout as torch.cat([h(x) for h in heads], dim=-1).
        for name in ("query", "key", "value"):
            w = torch.cat([sd[f"{p}sa.heads.{h}.{name}.weight"] for h in range(n_head)], dim=0)
            tensors.append((f"{l}.w{name[0]}", w))
        tensors += [
            (f"{l}.wo", sd[f"{p}sa.proj.weight"]), (f"{l}.bo", sd[f"{p}sa.proj.bias"]),
            (f"{l}.ln1_w", sd[f"{p}ln1.weight"]), (f"{l}.ln1_b", sd[f"{p}ln1.bias"]),
            (f"{l}.ln2_w", sd[f"{p}ln2.weight"]), (f"{l}.ln2_b", sd[f"{p}ln2.bias"]),
            (f"{l}.w1", sd[f"{p}ff.fc.weight"]), (f"{l}.b1", sd[f"{p}ff.fc.bias"]),
            (f"{l}.w2", sd[f"{p}ff.proj.weight"]), (f"{l}.b2", sd[f"{p}ff.proj.bias"]),
        ]
    tensors += [
        ("lnf_w", sd["ln_f.weight"]), ("lnf_b", sd["ln_f.bias"]),
        ("lm_w", sd["lm_head.weight"][kept_t]), ("lm_b", sd["lm_head.bias"][kept_t]),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries, offset, chunks = [], 0, []
    for name, t in tensors:
        arr = t.contiguous().numpy().astype("<f2")
        entries.append({"name": name, "shape": list(t.shape), "offset": offset})
        offset += arr.size
        chunks.append(arr.tobytes())
    (OUT_DIR / "weights.bin").write_bytes(b"".join(chunks))

    n_trained = model.num_params()
    n_exported = sum(t.numel() for _, t in tensors)
    manifest = {
        "config": {k: cfg[k] for k in ("n_embd", "n_head", "n_layer", "block_size")},
        "vocab": kept.tolist(),
        "tensors": entries,
        "params_trained": n_trained,
        "params_exported": n_exported,
        "iter": ckpt["iter"],
        "val_loss": ckpt["val_loss"],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

    # Reference logits from the full fp32 PyTorch model, for check_parity.mjs.
    import tiktoken
    ids = tiktoken.get_encoding("gpt2").encode(FIXTURE_PROMPT)
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    dropped_mass = 1.0 - probs[kept_t].sum().item()
    # Worst case over every position of the prompt, not just the last one.
    all_probs = torch.softmax(logits[0], dim=-1)
    worst_dropped = (1.0 - all_probs[:, kept_t].sum(dim=-1)).max().item()
    FIXTURE_PATH.write_text(json.dumps({
        "prompt": FIXTURE_PROMPT,
        "ids": ids,
        "logits": last[kept_t].tolist(),
    }))

    print(f"checkpoint      : iter {ckpt['iter']} | val {ckpt['val_loss']:.4f}")
    print(f"vocab kept      : {len(kept):,} / {cfg['vocab_size']:,}")
    print(f"params          : {n_trained:,} trained -> {n_exported:,} exported")
    print(f"weights.bin     : {offset * 2 / 1e6:.2f} MB (fp16)")
    print(f"dropped p-mass  : {dropped_mass:.2e} (last pos), {worst_dropped:.2e} (worst pos)")


if __name__ == "__main__":
    main()
