#!/usr/bin/env python3
"""What can this GPU actually execute?

Every quantization path is tried on a real matmul and the outcome recorded --
including the failures, with their exact error. A refusal is a result: it is a
cell in the paper's hardware-feasibility table, measured rather than quoted
from a library's documentation.

    python probe_capability.py out.json          on a GPU node
    python probe_capability.py --dry-run         anywhere, ~1 second

--dry-run skips every GPU call and only asks whether this image's torchao
exposes each level's config class at all, reading the level table out of
run_llm.py so the two can never disagree. It also prints the torch, torchao and
bitsandbytes versions, which is the quickest way to tell two .sif files apart.

Five minutes at most, no model loading, no downloads.
"""
import json, os, sys, time

_TAKES_VALUE = {"--levels"}
_ARGS, _FLAGS = [], {}
_argv = sys.argv[1:]
_i = 0
while _i < len(_argv):
    _a = _argv[_i]
    if _a.startswith("--"):
        if "=" in _a:
            _k, _v = _a.split("=", 1)
            _FLAGS[_k] = _v
        elif (_a in _TAKES_VALUE and _i + 1 < len(_argv)
              and not _argv[_i + 1].startswith("--")):
            _FLAGS[_a] = _argv[_i + 1]
            _i += 1
        else:
            _FLAGS[_a] = ""
    else:
        _ARGS.append(_a)
    _i += 1
OUT = _ARGS[0] if _ARGS else "capability.json"
out = {}

# ---------------------------------------------------------------------------
#  --dry-run: "does this image have a path for this level at all?"
#
#  Everything below this block needs a GPU -- the first thing it does is
#  torch.cuda.get_device_capability(0) -- so on a login node the probe used to
#  die on its first line. But the question that decides whether a level is
#  worth queueing is not a GPU question: it is whether this image's torchao
#  exposes the config class at all, which is an import and a constructor.
#
#  So --dry-run answers that with no GPU, no model and no allocation, in about
#  a second, and can be run on the head node before anything is submitted.
#  What it CANNOT tell you is whether the kernels actually execute on this
#  card. Run the full probe on a GPU node for that.
#
#      python probe_capability.py --dry-run
#      python probe_capability.py --dry-run --levels=int2_ao,w2a8_intx
# ---------------------------------------------------------------------------
if "--dry-run" in _FLAGS:
    import os

    import torch

    def _runner_specs():
        """AO_SPECS and _build exactly as run_llm.py defines them.

        run_llm.py cannot be imported: at module level it reads QB_MODEL_DIR,
        QB_SUBSET and QB_OUT from the environment, opens the subset file and
        loads the model. So slice the definitions out of the source and exec
        only those -- the same trick check_their_aug.py uses for roots_of().

        The point is that this reports what the RUNNER will resolve, not a
        second copy of the table that can drift away from it. A probe that
        says a level is fine while run_llm.py refuses it is worse than no
        probe at all.
        """
        import ast
        here = os.path.dirname(os.path.abspath(__file__))
        src_path = None
        for cand in ("run_llm.py", "qb-run_llm.py"):
            if os.path.exists(os.path.join(here, cand)):
                src_path = os.path.join(here, cand)
                break
        if src_path is None:
            return None, None, ("run_llm.py (or qb-run_llm.py) is not beside "
                                "this script, so the level table cannot be read")
        src = open(src_path).read()
        tree = ast.parse(src)
        want_fn = {"_fp4", "_fp8", "_int2", "_uint2", "_per_group",
                   "resolve_ao_config", "_ao_version"}
        want_as = {"_AOQ", "_MX", "AO_SPECS", "MX_LEVELS", "STRICT_KWARGS",
                   "AO_LEVELS", "BNB"}
        chunks = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in want_fn:
                chunks.append(ast.get_source_segment(src, node))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in want_as:
                        chunks.append(ast.get_source_segment(src, node))
        ns = {"torch": torch}
        exec(compile("\n\n".join(chunks), "run_llm-slice", "exec"), ns)
        return ns, src_path, None

    out["mode"] = "dry-run"
    out["torch"] = torch.__version__
    out["torch_has_int2"] = hasattr(torch, "int2")
    out["torch_has_uint2"] = hasattr(torch, "uint2")
    try:
        import torchao
        out["torchao"] = torchao.__version__
    except Exception as e:
        out["torchao"] = f"unavailable: {type(e).__name__}: {e}"
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception as e:
        out["bitsandbytes"] = f"unavailable: {type(e).__name__}"

    print("DRY RUN -- no GPU touched, no model loaded, nothing allocated")
    print(f"   torch {out['torch']}   int2 dtype: "
          f"{'yes' if out['torch_has_int2'] else 'NO'}   uint2 dtype: "
          f"{'yes' if out['torch_has_uint2'] else 'NO'}")
    print(f"   torchao {out['torchao']}")
    print(f"   bitsandbytes {out['bitsandbytes']}")
    print()

    ns, src_path, err = _runner_specs()
    if err:
        print(f"   {err}")
        out["levels"] = {"error": err}
    else:
        print(f"   level table read from {src_path}")
        want = _FLAGS.get("--levels") or ""
        levels = [x for x in want.split(",") if x] or sorted(ns["AO_SPECS"])
        out["levels"] = {}
        print()
        print("   %-13s %-9s %s" % ("level", "status", "what resolved / why not"))
        print("   " + "-" * 84)
        for lvl in levels:
            if lvl in ns.get("BNB", {}):
                ok = not str(out["bitsandbytes"]).startswith("unavailable")
                detail = ("bitsandbytes %s" % out["bitsandbytes"] if ok
                          else str(out["bitsandbytes"]))
                out["levels"][lvl] = {"available": ok, "detail": detail}
                print("   %-13s %-9s %s" % (lvl, "OK" if ok else "MISSING", detail))
                continue
            if lvl not in ns["AO_SPECS"]:
                out["levels"][lvl] = {"available": False,
                                      "detail": "not a level this build knows"}
                print("   %-13s %-9s %s" % (lvl, "UNKNOWN",
                                            "not a level run_llm.py defines"))
                continue
            try:
                _cfg, where = ns["resolve_ao_config"](lvl)
                out["levels"][lvl] = {"available": True, "detail": where}
                print("   %-13s %-9s %s" % (lvl, "OK", where))
            except Exception as e:
                why = str(e).splitlines()
                # the first line is the verdict, the rest is what was tried
                out["levels"][lvl] = {"available": False,
                                      "detail": " | ".join(x.strip() for x in why)}
                print("   %-13s %-9s %s" % (lvl, "MISSING", why[0]))
                for extra in why[1:]:
                    print("   %-13s %-9s   %s" % ("", "", extra.strip()))

    json.dump(out, open(OUT, "w"), indent=2)
    print()
    print(f"   written: {OUT}")
    print()
    print("   A level marked OK here has the CLASS. Whether its kernels run on")
    print("   the card is a separate question -- run this without --dry-run on a")
    print("   GPU node to settle that.")
    sys.exit(0)

# ---------------------------------------------------------------------------
#  STAGED STARTUP.
#
#  This block used to be five bare calls. On an RTX 3070 under WSL with the
#  CUDA 13.2 image it produced "Segmentation fault (core dumped)" and NOTHING
#  else -- a segfault discards whatever is sitting in stdout's buffer, so the
#  lines that had already been printed were lost with it and there was no way
#  to tell which call died.
#
#  Every step now announces itself and flushes BEFORE doing the thing, so the
#  last line on screen names the call that crashed. The driver version is read
#  from nvidia-smi first, because the most likely cause of a hard crash at CUDA
#  init is a runtime newer than the driver -- and that is a fact about the
#  machine, not a bug to debug.
# ---------------------------------------------------------------------------
def step(msg):
    print(f"   .. {msg}", flush=True)


step("nvidia-smi (before torch touches the driver)")
try:
    import subprocess
    smi = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60)
    line = (smi.stdout or smi.stderr).strip().splitlines()
    out["nvidia_smi"] = line[0] if line else "(no output)"
    print(f"      {out['nvidia_smi']}", flush=True)
except Exception as e:
    out["nvidia_smi"] = f"unavailable: {type(e).__name__}: {e}"
    print(f"      unavailable: {type(e).__name__}", flush=True)

step("import torch")
import torch
import torch.nn as nn
out["torch"] = torch.__version__
out["torch_cuda_build"] = torch.version.cuda
print(f"      torch {torch.__version__}  built for CUDA {torch.version.cuda}",
      flush=True)

# ---------------------------------------------------------------------------
#  WHICH libcuda IS THE LOADER GOING TO USE?
#
#  MEASURED 2026-09-16: an RTX 3070 Laptop under WSL, driver 610.88 -- far
#  newer than the CUDA 13.2 this image is built for -- segfaulted inside
#  torch.cuda.is_available(). A driver that new rules out the obvious cause, so
#  the remaining suspect is which libcuda.so.1 wins.
#
#  On WSL the ONLY usable libcuda is the host's, under /usr/lib/wsl/lib. NVIDIA
#  CUDA base images also ship /usr/local/cuda/compat/libcuda.so.*, which exists
#  precisely to override the host driver on normal Linux -- and on WSL that
#  override is fatal, because the WSL stub talks to /dev/dxg and the compat
#  library does not. If compat wins the search order, this is exactly what you
#  get: a hard crash at the first driver call, with no Python-level error.
#
#  None of this touches CUDA, so it is safe to run before the call that dies.
# ---------------------------------------------------------------------------
step("library resolution (nothing here touches CUDA)")
try:
    import glob as _glob
    import subprocess as _sp
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    out["ld_library_path"] = ld
    print(f"      LD_LIBRARY_PATH = {ld or '(empty)'}", flush=True)
    compat = sorted(_glob.glob("/usr/local/cuda*/compat/libcuda.so*"))
    out["cuda_compat"] = compat
    print("      cuda compat libs: %s"
          % (", ".join(compat) if compat else "none (good on WSL)"), flush=True)
    wsl = sorted(_glob.glob("/usr/lib/wsl/lib/libcuda.so*"))
    out["wsl_libcuda"] = wsl
    print("      /usr/lib/wsl/lib: %s"
          % (", ".join(os.path.basename(x) for x in wsl) if wsl
             else "ABSENT -- the host driver is not bound into the container"),
          flush=True)
    out["dev_dxg"] = os.path.exists("/dev/dxg")
    print("      /dev/dxg present : %s   (WSL GPU node)" % out["dev_dxg"],
          flush=True)
    try:
        r = _sp.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=60)
        cand = [l.strip() for l in r.stdout.splitlines() if "libcuda.so" in l]
        out["ldconfig_libcuda"] = cand[:6]
        for c in cand[:6]:
            print(f"      ldconfig: {c}", flush=True)
        if not cand:
            print("      ldconfig: no libcuda.so entry", flush=True)
    except Exception as _e:
        print(f"      ldconfig unavailable: {type(_e).__name__}", flush=True)
    if compat and wsl:
        print("      !! BOTH a compat libcuda and the WSL one are visible. On WSL",
              flush=True)
        print("         the compat one must NOT win. That is the likely crash.",
              flush=True)
except Exception as e:
    out["library_resolution"] = f"{type(e).__name__}: {e}"
    print(f"      check failed: {type(e).__name__}", flush=True)

step("torch.cuda.is_available()")
_avail = torch.cuda.is_available()
print(f"      {_avail}", flush=True)
if not _avail:
    out["fatal"] = "torch.cuda.is_available() is False"
    print("\n   No usable GPU inside the container. Under WSL this is almost\n"
          "   always the driver: the image is built for CUDA "
          f"{torch.version.cuda} and needs a\n"
          "   host driver at least that new. Check the driver_version above.\n"
          "   Run this on a cluster node instead, or use --dry-run, which needs\n"
          "   no GPU at all.", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    sys.exit(2)

step("torch.cuda.device_count()")
out["n_gpus"] = torch.cuda.device_count()
print(f"      {out['n_gpus']}", flush=True)

step("torch.cuda.get_device_capability(0)")
cc = torch.cuda.get_device_capability(0)
out["sm"] = cc[0] * 10 + cc[1]
print(f"      sm_{out['sm']}", flush=True)

step("torch.cuda.get_device_properties(0)")
props = torch.cuda.get_device_properties(0)
out["gpu"] = props.name
out["vram_gb"] = round(props.total_memory / 1024**3, 1)
print(f"      {out['gpu']}  {out['vram_gb']} GB", flush=True)

step("first allocation on the device")
_t = torch.zeros(1024, device="cuda")
torch.cuda.synchronize()
del _t
print("      ok", flush=True)

print(f"   {out['gpu']}  sm_{out['sm']}  {out['vram_gb']} GB  "
      f"torch {out['torch']}  ({out['n_gpus']} visible)", flush=True)

# dgl's compiled extension dlopens libcuda.so.1, which does not exist during a
# docker build -- so the image can only check the .so shipped. With a driver
# present we do the real import here; LocalRetro and RetroKNN depend on it.
step("import dgl (Syntheseus members only; absent from the LLM image)")
try:
    import dgl, dgl.graphbolt
    out["dgl"] = {"version": dgl.__version__, "graphbolt": True}
    print(f"   dgl {dgl.__version__} + graphbolt loaded")
except Exception as e:
    out["dgl"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    print(f"   dgl unusable: {type(e).__name__} -- localretro/retroknn will fail")


def _per_group_probe(n):
    """PerGroup moved between torchao releases; try both homes."""
    try:
        from torchao.quantization import PerGroup
    except ImportError:
        from torchao.quantization.granularity import PerGroup
    return PerGroup(n)


def bench(label, fn, warmup=1, iters=10):
    """Time after a warm-up, so the first-call cost is not reported as speed."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / iters
        ok = bool(torch.isfinite(y).all().item())
        out[label] = {"ok": ok, "ms": round(ms, 3)}
        print(f"   {'yes' if ok else ' no'}  {label:24s} {ms:9.3f} ms")
    except Exception as e:
        out[label] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"    no  {label:24s} {type(e).__name__}: {str(e)[:100]}")


def dtype_case(dt):
    def f():
        net = nn.Linear(1024, 1024, bias=False).to(dt).cuda()
        x = torch.randn(256, 1024, device="cuda", dtype=dt)
        with torch.no_grad():
            return net(x)
    return f


bench("fp32", dtype_case(torch.float32))
bench("fp16", dtype_case(torch.float16))
bench("bf16", dtype_case(torch.bfloat16))   # bf16 tensor cores start with Ampere

# MSLK supplies the kernels behind int4-plain, fp8_w4a8 and dynamic NVFP4. Its
# presence or absence decides three rows below, so record it explicitly rather
# than leaving the reader to infer it from three identical ImportErrors.
try:
    import mslk
    out["mslk"] = getattr(mslk, "__version__", "present")
    print(f"   mslk {out['mslk']}")
except Exception as e:
    out["mslk"] = f"absent: {type(e).__name__}"
    print("   mslk ABSENT -- int4_weight_only_mslk, fp8_w4a8 and nvfp4 will fail")
    print("     install: pip install mslk --index-url "
          "https://download.pytorch.org/whl/cu130")

try:
    import torchao
    import torchao.quantization as Q
    out["torchao"] = torchao.__version__
    print(f"   torchao {torchao.__version__}")

    # Each level names the MODULE its config must come from. hasattr(Q, name)
    # was not enough: torchao 0.18 moved Int8DynamicActivationInt4WeightConfig
    # into torchao.prototype.quantization.int4 and made it a CPU backend, so a
    # name-only lookup can find a class that answers to "w4a8" and computes on
    # the CPU. The Blackwell formats live in torchao.prototype.mx_formats and
    # are never reachable from torchao.quantization at all.
    import importlib

    def ao_case(targets, dt=torch.float32, kw=None):
        def f():
            net = nn.Sequential(nn.Linear(1024, 1024, bias=False)).to(dt).cuda()
            cfg, tried = None, []
            for modname, attr in targets:
                try:
                    mod = importlib.import_module(modname)
                except Exception as e:
                    tried.append(f"{modname} import {type(e).__name__}")
                    continue
                o = getattr(mod, attr, None)
                if o is None:
                    tried.append(f"{modname}.{attr} absent")
                    continue
                try:
                    cfg = o(**(kw() if kw else {}))
                except TypeError:
                    cfg = o()      # older torchao lacking those fields
                break
            if cfg is None:
                raise AttributeError("; ".join(tried))
            Q.quantize_(net, cfg)
            x = torch.randn(256, 1024, device="cuda", dtype=dt)
            with torch.no_grad():
                return net(x)
        return f

    AOQ, MX = "torchao.quantization", "torchao.prototype.mx_formats"
    BF = torch.bfloat16

    bench("int8_weight_only", ao_case([(AOQ, "Int8WeightOnlyConfig"),
                                       (AOQ, "int8_weight_only")]))
    bench("w8a8", ao_case([(AOQ, "Int8DynamicActivationInt8WeightConfig"),
                           (AOQ, "int8_dynamic_activation_int8_weight")]))
    # Two rows, because they are different kernels and only one of them is
    # available without the separate mslk package. torchao 0.18 made "plain"
    # (mslk) the default; tile_packed_to_4d is the tinygemm path that torchao
    # 0.9.0 used for the published numbers.
    bench("int4_weight_only_mslk", ao_case([(AOQ, "Int4WeightOnlyConfig"),
                                            (AOQ, "int4_weight_only")], BF))
    bench("int4_weight_only_tinygemm",
          ao_case([(AOQ, "Int4WeightOnlyConfig")], BF,
                  lambda: {"int4_packing_format": "tile_packed_to_4d"}))
    bench("w4a8", ao_case([(AOQ, "Int8DynamicActivationInt4WeightConfig"),
                           (AOQ, "int8_dynamic_activation_int4_weight")], BF))
    bench("w4a4", ao_case([(AOQ, "Int4DynamicActivationInt4WeightConfig"),
                           (AOQ, "int4_dynamic_activation_int4_weight")], BF))
    # w4a8 rebuilt from the general intx config for torchao >= 0.14, where the
    # dedicated class no longer exists. Same recipe, different kernel, so it is
    # timed as its own row rather than folded into w4a8.
    bench("w4a8_intx",
          ao_case([(AOQ, "Int8DynamicActivationIntxWeightConfig")], BF,
                  lambda: {"weight_dtype": torch.int4}))
    # 8-bit float: Ada (sm_89) and newer.
    bench("fp8_weight_only", ao_case([(AOQ, "Float8WeightOnlyConfig")], BF))
    bench("fp8_w8a8",
          ao_case([(AOQ, "Float8DynamicActivationFloat8WeightConfig")], BF))
    bench("fp8_w4a8",
          ao_case([(AOQ, "Float8DynamicActivationInt4WeightConfig")], BF))
    # 4-bit float: Blackwell. torchao gates these on compute capability >= 10.0,
    # so an RTX 5090 (12.0) clears the gate and a 3090 (8.6) does not. Whether
    # the kernels were compiled for sm_120 in THIS wheel is what the timing
    # below actually settles.
    bench("nvfp4_weight_only", ao_case([(MX, "NVFP4WeightOnlyConfig")], BF))
    bench("nvfp4",
          ao_case([(MX, "NVFP4DynamicActivationNVFP4WeightConfig")], BF))
    bench("mxfp8", ao_case([(MX, "MXDynamicActivationMXWeightConfig")], BF,
                           lambda: {"activation_dtype": torch.float8_e4m3fn,
                                    "weight_dtype": torch.float8_e4m3fn}))
    bench("mxfp4", ao_case([(MX, "MXDynamicActivationMXWeightConfig")], BF,
                           lambda: {"activation_dtype": torch.float4_e2m1fn_x2,
                                    "weight_dtype": torch.float4_e2m1fn_x2}))
    # 2-bit. Two spellings: torchao >= 0.14 has IntxWeightOnlyConfig and has
    # dropped UIntXWeightOnlyConfig; 0.9.0 has only the latter. Both are timed
    # so the capability table records WHICH one this image answered to -- the
    # run_llm.py level resolves the same way and must not disagree with this.
    # A row that appears here but is no faster than bf16 is expected, not a
    # fault: the portable intx path dequantizes to bf16 for the matmul.
    def _t(name):
        return getattr(torch, name, None)

    if _t("int2") is not None:
        bench("int2_weight_only_intx",
              ao_case([(AOQ, "IntxWeightOnlyConfig")], BF,
                      lambda: {"weight_dtype": torch.int2,
                               "granularity": _per_group_probe(32)}))
        bench("w2a8_intx",
              ao_case([(AOQ, "Int8DynamicActivationIntxWeightConfig")], BF,
                      lambda: {"weight_dtype": torch.int2}))
    else:
        out["int2_weight_only_intx"] = {"error": "torch has no int2 dtype "
                                        f"(torch {torch.__version__} < 2.6)"}
    if _t("uint2") is not None:
        bench("int2_weight_only_uintx",
              ao_case([(AOQ, "UIntXWeightOnlyConfig")], BF,
                      lambda: {"dtype": torch.uint2, "group_size": 32}))
except Exception as e:
    out["torchao"] = f"unavailable: {type(e).__name__}"
    print(f"   torchao unavailable: {e}")

# bitsandbytes 4-bit. Needs sm_75, and is the only route to NF4 -- worth
# recording as a row rather than discovering at hour six of a run.
try:
    import bitsandbytes as bnb
    out["bitsandbytes"] = bnb.__version__
    print(f"   bitsandbytes {bnb.__version__}")

    def bnb_case(qt):
        def f():
            lin = bnb.nn.Linear4bit(1024, 1024, bias=False, quant_type=qt,
                                    compute_dtype=torch.bfloat16).cuda()
            x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                return lin(x)
        return f

    bench("bnb_nf4", bnb_case("nf4"))
    bench("bnb_fp4", bnb_case("fp4"))
except Exception as e:
    out["bitsandbytes"] = f"unavailable: {type(e).__name__}"
    print(f"   bitsandbytes unavailable: {type(e).__name__} -- no NF4/FP4")

# --------------------------------------------------------------------------- #
#  BYTES PER WEIGHT -- does this level actually save VRAM?
#
#  The level NAME does not tell you. torchao's portable intx packing is
#  IntxPackingFormat.UNPACKED_TO_INT8: one sub-byte value stored per int8 byte,
#  plus a scale and zero-point per group, dequantized to bf16 for the matmul.
#  So "4-bit" w4a8_intx was MEASURED at 9.12 GB on the 8B model against int8's
#  8.22 GB -- more memory for fewer bits -- while bitsandbytes nf4, which packs
#  for real, came in at 5.99 GB.
#
#  This measures the allocator directly on one large Linear, so it costs a few
#  hundred MB and a second, and it settles the question on any card including a
#  laptop that cannot hold the 8B model at all. 2.0 bytes/weight is bf16; below
#  1.0 means real packing; at or above 1.0 means the level is an ACCURACY
#  experiment and must stay out of the memory table.
# --------------------------------------------------------------------------- #
N_MEM = 4096


def _resolve(targets, kw):
    tried = []
    for modname, attr in targets:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            tried.append(f"{modname} import {type(e).__name__}")
            continue
        o = getattr(mod, attr, None)
        if o is None:
            tried.append(f"{modname}.{attr} absent")
            continue
        try:
            return o(**(kw() if kw else {}))
        except TypeError as e:
            # Deliberately NOT retrying bare here. For int2/int4/MX the kwargs
            # carry the bit width, and the bare constructor returns a DIFFERENT
            # width -- measuring that and printing it under this label would be
            # a fabricated row.
            tried.append(f"{modname}.{attr} rejected kwargs ({e})")
    raise AttributeError("; ".join(tried) or "no candidate")


def mem_of(label, build):
    """Bytes resident per weight after quantizing one N_MEM x N_MEM Linear."""
    try:
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        net = build()
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - base
        bpw = held / float(N_MEM * N_MEM)
        del net
        _gc.collect()
        torch.cuda.empty_cache()
        out.setdefault("bytes_per_weight", {})[label] = round(bpw, 4)
        verdict = ("packs" if bpw < 1.0 else
                   "NO saving" if bpw < 2.0 else "same as bf16")
        print(f"   {label:24s} {bpw:6.3f} bytes/weight   {verdict}")
    except Exception as e:
        out.setdefault("bytes_per_weight", {})[label] = \
            f"unavailable: {type(e).__name__}: {str(e)[:120]}"
        print(f"   {label:24s} {type(e).__name__}: {str(e)[:70]}")


print()
print("   bytes per weight (one %d x %d Linear, measured from the allocator)"
      % (N_MEM, N_MEM))


def _plain(dt):
    return lambda: nn.Linear(N_MEM, N_MEM, bias=False).to(dt).cuda()


mem_of("bf16 (reference)", _plain(torch.bfloat16))
try:
    import torchao.quantization as Q
    import importlib
    AOQ = "torchao.quantization"

    def _ao(targets, kw=None):
        def b():
            net = nn.Linear(N_MEM, N_MEM, bias=False).to(torch.bfloat16).cuda()
            Q.quantize_(net, _resolve(targets, kw))
            return net
        return b

    mem_of("int8_ao", _ao([(AOQ, "Int8WeightOnlyConfig")]))
    def _ao_or_bare(targets, kw):
        """kwargs first, then bare. Only for levels where the bare constructor
        yields the SAME bit width -- int4_ao's kwarg picks the packing format,
        not the width. torchao 0.13's Int4WeightOnlyConfig has no
        int4_packing_format field, so without this the row read
        'AttributeError: rejected kwargs' and the one packed torchao integer
        level was missing from the comparison."""
        def b():
            net = nn.Linear(N_MEM, N_MEM, bias=False).to(torch.bfloat16).cuda()
            try:
                cfg = _resolve(targets, kw)
            except Exception:
                cfg = _resolve(targets, None)
            Q.quantize_(net, cfg)
            return net
        return b

    mem_of("int4_ao", _ao_or_bare([(AOQ, "Int4WeightOnlyConfig")],
                                  lambda: {"int4_packing_format": "tile_packed_to_4d"}))
    mem_of("w4a8_intx", _ao([(AOQ, "Int8DynamicActivationIntxWeightConfig")],
                            lambda: {"weight_dtype": torch.int4}))
    if hasattr(torch, "int2"):
        mem_of("int2_ao", _ao([(AOQ, "IntxWeightOnlyConfig")],
                              lambda: {"weight_dtype": torch.int2,
                                       "granularity": _per_group_probe(32)}))
        mem_of("w2a8_intx", _ao([(AOQ, "Int8DynamicActivationIntxWeightConfig")],
                                lambda: {"weight_dtype": torch.int2}))
except Exception as e:
    print(f"   torchao memory rows skipped: {type(e).__name__}")
try:
    import bitsandbytes as _bnb

    def _bnb_lin(qt):
        return lambda: _bnb.nn.Linear4bit(
            N_MEM, N_MEM, bias=False, quant_type=qt,
            compute_dtype=torch.bfloat16).cuda()

    mem_of("bnb_nf4", _bnb_lin("nf4"))
    mem_of("bnb_fp4", _bnb_lin("fp4"))
except Exception as e:
    print(f"   bitsandbytes memory rows skipped: {type(e).__name__}")

# --------------------------------------------------------------------------- #
#  SUMMARY. The rows above are individually informative and collectively look
#  like a wall of failure, which they are not: a refusal is a result. What
#  matters is WHICH KIND of refusal, because the two have opposite remedies.
#
#     build     the operator is missing from this wheel. Another wheel, or an
#               extra package, may supply it. Changing GPU will not.
#     hardware  this architecture cannot execute it. Only a different GPU will.
# --------------------------------------------------------------------------- #
BUILD_MARKS = ("absent", "requires mslk", "mslk is required", "importerror",
               "no module named", "not available for build")
HW_MARKS = ("compute capability", "sm100", "sm_100", "cuda>=8.9",
            "named symbol not found", "no kernel image",
            "b200", "b300",   # MXFP4 is gated on datacenter Blackwell
            # Measured on an RTX 2080 Ti (sm_75), where both of these landed in
            # UNCLASSIFIED and had to be diagnosed by hand:
            #   int4_weight_only_mslk -> "cutlass cannot initialize", preceded by
            #     pages of "Failed to initialize the TMA descriptor". TMA is the
            #     Tensor Memory Accelerator, a Hopper (sm_90) unit. MSLK's int4
            #     kernels require it, so this is an architecture limit and will
            #     never be fixed by rebuilding the image.
            #   fp8_w4a8 -> "type fp8e4nv not supported in this architecture",
            #     raised by Triton. FP8 arithmetic needs sm_89+.
            "cutlass cannot initialize", "tma descriptor",
            "not supported in this architecture", "fp8e4nv")

usable, by_build, by_hw, other = [], [], [], []
for name, rec in out.items():
    if not isinstance(rec, dict) or "ok" not in rec:
        continue                      # gpu/sm/vram/dgl metadata, not a level
    if rec.get("ok"):
        usable.append(f"{name} ({rec.get('ms', 0):.1f} ms)")
        continue
    e = str(rec.get("err", "")).lower()
    if any(m in e for m in BUILD_MARKS):
        by_build.append(name)
    elif any(m in e for m in HW_MARKS):
        by_hw.append(name)
    else:
        other.append(name)

out["summary"] = {"usable": usable, "missing_from_build": by_build,
                  "unsupported_by_architecture": by_hw, "other": other}

print(f"\n   ---- summary for {out['gpu']} (sm_{out['sm']}) ----")


def show(label, items, note):
    if not items:
        return
    print(f"   {label} ({len(items)})")
    for i in items:
        print(f"       {i}")
    print(f"       -> {note}")


show("RUNS HERE", usable, "measurable on this card")
show("MISSING FROM THIS BUILD", by_build,
     "a packaging gap: the SAME failure on every GPU, including Blackwell")
show("NOT SUPPORTED BY THIS ARCHITECTURE", by_hw,
     "needs a newer GPU; expected to clear on sm_100+")
show("UNCLASSIFIED", other, "read the error above")

print("\n   Caveat: each row is a 1024x1024 linear layer, so it answers 'can")
print("   this card execute this format', NOT 'does the 8B model fit'. On a")
print(f"   {out['vram_gb']:.0f} GB card the second question is the binding one.")

json.dump(out, open(OUT, "w"), indent=2)
print(f"\n   written {OUT}")
print("   Integer tensor cores start with Turing (sm_75), and libraries differ")
print("   on where they draw the line. The rows above are what this card")
print("   does, not what documentation claims.")
