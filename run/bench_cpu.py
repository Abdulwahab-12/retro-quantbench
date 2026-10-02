#!/usr/bin/env python3
"""Can this run on CPU, how slow, and does quantization help there?

    bash qb.sh --bench-cpu

WHY A LAYER AND NOT THE WHOLE MODEL
Loading 16 GB into RAM takes minutes and would have to be repeated per format,
so a full-model sweep costs an hour to answer a question that one layer answers
in seconds. CPU decoding of an 8B model is MEMORY-BANDWIDTH bound: every
generated token requires reading every weight once. So the quantity that decides
everything is bytes-per-second through the memory system, and a single large
matmul in the model's real FFN shape (4096 x 12288) measures exactly that.

The extrapolation at the end converts GB/s into tokens/s for the full model,
which is the number the question is actually about.

WHY QUANTIZATION CAN HELP ON CPU, AND WHEN IT DOES NOT
On a GPU, 4-bit weights mostly buy memory capacity. On a CPU they buy TIME,
because halving the bytes halves the traffic that the bottleneck is made of --
but ONLY if an optimised kernel exists for that format. Without one, PyTorch
dequantises to fp32 and does the same matmul it would have done anyway, having
paid extra to unpack. That case is SLOWER than not quantizing, and it is why
this script measures rather than assumes.

bitsandbytes nf4/fp4 are CUDA kernels and are skipped here on purpose.
"""
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch

torch.set_grad_enabled(False)

# The model's real feed-forward shape (Qwen3-8B: d_model 4096, d_ffn 12288).
# A 1024x1024 toy layer sits in cache and measures nothing useful.
D_IN, D_OUT = 4096, 12288
REPS = int(os.environ.get("QB_CPU_REPS", "5"))
N_PARAMS = 8.19e9          # RetroDFM-R-8B
N_LAYERS = 36


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


print("=" * 66)
print("CPU")
print("=" * 66)
print(f"  {cpu_name()}")
print(f"  cores (torch)  : {torch.get_num_threads()} threads of "
      f"{os.cpu_count()} logical")
print(f"  RAM            : {ram_gb():.0f} GB")
print(f"  layer under test: {D_IN} x {D_OUT}, {REPS} repetitions\n")

x = torch.randn(1, D_IN, dtype=torch.float32)


def bench(name, make, dtype_bytes, in_dtype):
    """Time one linear layer; report achieved bandwidth.

    in_dtype is passed in rather than read off the weight. After quantize_ the
    weight is a tensor subclass whose .dtype may report int8, and feeding an
    int8 activation into the layer is both wrong and an instant exception --
    the activation stays bf16/fp32 regardless of how the weight is stored.
    """
    try:
        lin = make()
    except Exception as e:
        return name, None, None, f"{type(e).__name__}: {e}".replace("\n", " ")[:44]
    xx = x.to(in_dtype)
    try:
        lin(xx)                                    # warm up / allocate
        t0 = time.perf_counter()
        for _ in range(REPS):
            lin(xx)
        dt = (time.perf_counter() - t0) / REPS
    except Exception as e:
        return name, None, None, f"{type(e).__name__}: {e}".replace("\n", " ")[:44]
    gb = D_IN * D_OUT * dtype_bytes / 2**30
    return name, dt, gb / dt, None


def fp(dt):
    return torch.nn.Linear(D_IN, D_OUT, bias=False).to(dt)


# (label, factory, bytes-per-weight, activation dtype)
CASES = [("fp32", lambda: fp(torch.float32), 4, torch.float32),
         ("bf16", lambda: fp(torch.bfloat16), 2, torch.bfloat16)]

try:
    from torchao.quantization import quantize_
    import torchao.quantization as Q

    def ao(cfg_name, kwargs=None):
        def make():
            lin = fp(torch.bfloat16)
            cfg = getattr(Q, cfg_name)(**(kwargs or {}))
            quantize_(lin, cfg)
            return lin
        return make

    CASES += [
        ("int8_weight_only  (torchao)", ao("Int8WeightOnlyConfig"), 1,
         torch.bfloat16),
        ("int8 dyn-act      (torchao)",
         ao("Int8DynamicActivationInt8WeightConfig"), 1, torch.bfloat16),
    ]

    # int4 with int8 activations. In torchao 0.18 this config is the one with a
    # real CPU backend (da8w4_linear_cpu), so it is the most likely format to
    # actually go FASTER on a CPU rather than merely smaller.
    def make_int4():
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig
        lin = fp(torch.bfloat16)
        quantize_(lin, Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4))
        return lin

    CASES.append(("int4 dyn-act8     (torchao)", make_int4, 0.5,
                  torch.bfloat16))
except Exception as _e:
    # Deliberately broad. torchao is optional here, and importing it can fail
    # for reasons that are not ImportError -- a version mismatch against torch
    # raises AttributeError inside torchao's own __init__, for instance. An
    # optional extra must never take the whole measurement down with it.
    print(f"  torchao unusable ({type(_e).__name__}: {_e}) "
          f"-- only fp32/bf16 measured\n")

print("=" * 66)
print("ONE LINEAR LAYER ON CPU")
print("=" * 66)
print(f"  {'format':30s} {'ms':>8s} {'GB/s':>8s} {'vs fp32':>9s}")
base = None
rows = []
for name, make, nbytes, in_dt in CASES:
    nm, dt, bw, err = bench(name, make, nbytes, in_dt)
    if err:
        print(f"  {nm:30s} {'--':>8s} {'--':>8s} {'':>9s}  {err}")
        continue
    if base is None:
        base = dt
    rows.append((nm, dt, bw, nbytes))
    print(f"  {nm:30s} {dt*1000:8.1f} {bw:8.2f} {base/dt:8.2f}x")

if not rows:
    sys.exit("no format ran")

print()
print("=" * 66)
print("EXTRAPOLATED TO THE FULL 8B MODEL")
print("=" * 66)
print("  Every generated token reads every weight once, so")
print("      tokens/s  =  achieved bandwidth / model size")
print()
print(f"  {'format':30s} {'model GB':>9s} {'tok/s':>8s} {'s/molecule*':>12s}")
for nm, dt, bw, nbytes in rows:
    model_gb = N_PARAMS * nbytes / 2**30
    tps = bw / model_gb
    print(f"  {nm:30s} {model_gb:9.1f} {tps:8.2f} {335/tps:12.0f}")
print("\n  * one un-augmented molecule is about 335 generated tokens.")
print("    At k_a=20 k_s=10 k_b=10 multiply by roughly 350.")
print("""
READING THIS
  If a quantized row shows HIGHER GB/s than bf16, the format has a real CPU
  kernel and quantization genuinely speeds up CPU inference. If it shows lower,
  PyTorch is dequantising back to fp32 before the matmul and you are paying for
  the unpacking with no benefit -- on that format, on this machine, quantization
  makes CPU inference slower, not faster.

  COMPARING AGAINST A GPU -- match the precision or say that you did not.

    same precision, valid:
      CPU bf16   2.0 tok/s   vs   RTX 5090 bf16   553 tok/s   = 275x

    NOT a like-for-like comparison:
      CPU bf16   2.0 tok/s   vs   RTX 2080 Ti nf4  30 tok/s
      The 2080 Ti cannot hold bf16 (15.3 GB of weights, 10.6 GB of card) and
      the CPU cannot run nf4 (bitsandbytes is CUDA-only). There is no shared
      precision, so 15x is "best each device can manage", not a speed ratio.
      That is still the useful number for planning -- just label it.""")
