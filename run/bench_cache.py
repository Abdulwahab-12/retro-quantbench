#!/usr/bin/env python3
"""Measure what KV-cache offloading actually costs, on THIS card.

    singularity exec --nv --bind $PWD:/work <image>.sif \
        python /opt/qb/scripts/bench_cache.py

Answers two questions with one short run, instead of estimating:

  1. How much GPU memory does each cache strategy need?
  2. How much slower is it?

WHY THIS EXISTS
The KV cache -- the stored Key and Value vectors for every token already
generated -- is the largest block of memory in an augmented run, and unlike
the weights it does NOT shrink when the model is quantized. At k_b=20 it is
about 4 GB, which is what excludes 8 GB cards from the 10x10x20 setting.

transformers can keep that cache in CPU RAM instead, moving each layer to the
GPU as it is needed. That trades PCIe bandwidth for VRAM. Arithmetic says the
whole cache crosses the bus once per generated token, so the penalty should be
large -- roughly 20x by a back-of-envelope estimate. An estimate is not a
measurement, and this script replaces it with one.

It runs ONE beam search per strategy, not a whole molecule, so it costs
minutes rather than hours.

Environment (all optional):
    QB_MODEL_DIR   weights            default /opt/model/retrodfm-r-8b
    QB_LEVEL       nf4 | fp4 | bf16   default nf4
    QB_KB          beams              default 20
    QB_NEW         tokens to generate default 128
    QB_PREFIX      prompt repetitions default 200 (~a realistic <think> length)
"""
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = os.environ.get("QB_MODEL_DIR", "/opt/model/retrodfm-r-8b")
LEVEL = os.environ.get("QB_LEVEL", "nf4")
KB = int(os.environ.get("QB_KB", "20"))
NEW = int(os.environ.get("QB_NEW", "128"))
PREFIX = int(os.environ.get("QB_PREFIX", "200"))

if not torch.cuda.is_available():
    print("no GPU visible", file=sys.stderr)
    sys.exit(1)

card = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"card   : {card}  {total:.1f} GB")
print(f"level  : {LEVEL}   beams: {KB}   new tokens: {NEW}\n")

kw = {"dtype": torch.bfloat16, "device_map": {"": 0}}
if LEVEL in ("nf4", "fp4"):
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=LEVEL,
        bnb_4bit_compute_dtype=torch.bfloat16)

print("loading ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
net = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, **kw)
net.eval()
weights = torch.cuda.memory_allocated() / 1024**3
print(f"weights resident: {weights:.2f} GB\n")

# A prompt long enough to build a realistic cache. The real runs beam-search
# from the end of a ~1200-token reasoning trace, so a short prompt would make
# offloading look far cheaper than it is.
text = "The product molecule is CCO. " * PREFIX
ids = tok(text, return_tensors="pt").to("cuda")
n_prompt = ids["input_ids"].shape[1]
print(f"prompt: {n_prompt} tokens (cache will hold {n_prompt + NEW} per beam)\n")

STRATEGIES = [
    ("default (all on GPU)", {}),
    ("offloaded to CPU", {"cache_implementation": "offloaded"}),
    ("quantized int4", {"cache_implementation": "quantized",
                        "cache_config": {"nbits": 4, "backend": "quanto"}}),
]

print(f"{'strategy':24s} {'peak GPU':>10s} {'cache':>8s} "
      f"{'seconds':>9s} {'tok/s':>8s} {'vs default':>11s}")
print("-" * 76)

base = None
for name, extra in STRATEGIES:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1234)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            net.generate(**ids, max_new_tokens=NEW, do_sample=False,
                         num_beams=KB, num_return_sequences=KB,
                         early_stopping=True,
                         pad_token_id=tok.eos_token_id, **extra)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        if base is None:
            base = dt
        rel = f"{dt/base:.1f}x" if base else "-"
        print(f"{name:24s} {peak:9.2f}G {peak-weights:7.2f}G "
              f"{dt:9.1f} {KB*NEW/dt:8.1f} {rel:>11s}")
    except torch.cuda.OutOfMemoryError:
        print(f"{name:24s} {'OOM':>10s} {'-':>8s} {'-':>9s} {'-':>8s} {'-':>11s}")
        torch.cuda.empty_cache()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:40]
        print(f"{name:24s} {msg}")
        torch.cuda.empty_cache()

print("""
Reading this table:
  'cache' is peak minus weights -- the part that scales with beams and
  sequence length, and the part that offloading moves off the card.
  A strategy that halves 'cache' but triples 'seconds' is only worth it
  if the card cannot otherwise hold the run at all.

  This is ONE beam search. A full molecule at k_a=10 k_s=10 runs 100 of
  them plus the sampling stage, so multiply 'seconds' accordingly.""")
