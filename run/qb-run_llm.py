#!/usr/bin/env python3
"""RetroDFM-R-8B inference worker: one shard of molecules, one GPU, one precision.

Writes the same JSONL schema every other member uses, so scoring, fidelity and
the ensemble frontier all consume it unchanged.

Driven entirely by environment variables (the cluster runner sets them):
    QB_MODEL_DIR  local HuggingFace weights
    QB_LEVEL      fp16 | bf16                          full precision
                  int8_ao | w8a8 | int4_ao | w4a8 | w4a4    torchao integer
                  fp8_ao | fp8_w8a8 | fp8_w4a8         torchao 8-bit float
                  nvfp4 | nvfp4_w | mxfp8 | mxfp4      torchao 4-bit float,
                                                       Blackwell only
                  nf4 | nf4dq | fp4                    bitsandbytes 4-bit
                  int2_ao | w2a8_intx                  torchao 2-bit -- read
                                                       the TWO-BIT note first
    QB_N          molecules from the subset
    QB_SUBSET     subset jsonl
    QB_SHARD      "i/N" -- interleaved slice for this worker
    QB_OUT        output jsonl

Resumable: molecules already in QB_OUT are read back and skipped. Records are
flushed and fsync'd per molecule, because two multi-hour runs were previously
lost to output buffering.
"""
import gc, json, os, random, sys, time, traceback, zlib

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# ---------------------------------------------------------------------------
#  FRAGMENTATION, not exhaustion.
#
#  On an 11 GB card at k_a=10 k_s=10 k_b=20 the log fills with lines like
#
#     OOM while trying to allocate 612368384 bytes (free: 590938112, ...)
#     OOM while trying to allocate 817889280 bytes (free: 765001728, ...)
#
#  Short by 21 MB and 51 MB respectively. The memory EXISTS; it is not in one
#  contiguous block. Generation allocates and frees KV-cache blocks of
#  constantly changing size (each molecule has a different prompt and answer
#  length), which is the worst case for the default caching allocator: it
#  reserves fixed segments and cannot lend the tail of one to another.
#
#  expandable_segments lets a segment grow in place instead, so a run of
#  variable-size allocations reuses one arena. Every failure above is a
#  cudaFree-and-retry, and that retry -- not the arithmetic -- is why the
#  2080 Ti reports 2500 s/mol against 235 s/mol on the big card.
#
#  This changes ONLY where tensors are placed. Not one number differs, so a
#  run with it is directly comparable to a run without it.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_DIR = os.environ["QB_MODEL_DIR"]
LEVEL     = os.environ.get("QB_LEVEL", "fp16")
N         = int(os.environ.get("QB_N", "5005"))
SUBSET    = os.environ["QB_SUBSET"]
OUT       = os.environ["QB_OUT"]
SHARD     = os.environ.get("QB_SHARD", "0/1")
SEED      = int(os.environ.get("QB_SEED", "1234"))
# Sampling at temperature 1.0 makes every comparison a mixture of two effects:
# the precision change we care about, and the chaotic amplification of any tiny
# numeric difference through a 2000-token chain. Two ways to separate them:
#   QB_SEED   -- rerun the SAME precision with a different seed to measure the
#                noise floor. A precision difference is only meaningful above it.
#   QB_GREEDY -- decode greedily (no sampling). Then two runs of the same
#                precision are identical by construction, so every difference
#                between precisions is numerics and nothing else. This is the
#                clean instrument for fidelity; the sampled arm remains the one
#                comparable to the paper's published accuracy.
GREEDY    = os.environ.get("QB_GREEDY", "0") == "1"
BASE_DTYPE = os.environ.get("QB_BASE_DTYPE", "bf16")   # base for torchao levels
MAX_NEW   = int(os.environ.get("QB_MAX_NEW", "2048"))   # eval.sh: 2048, not 1024

# How many of the k_s sampled paths to generate in ONE call. 0 = all at once,
# which is what the runs so far did and what the published numbers come from.
#
# num_return_sequences=k_s puts k_s sequences x MAX_NEW tokens of KV cache on
# the card at the same instant. For this 8B model that is roughly 150 kB per
# token per sequence, so k_s=10 at 2048 tokens is about 3 GB ON TOP of the
# weights -- the difference between fitting in 11 GB and not.
#
# Splitting the request costs nothing in accuracy: the paths are independent
# samples and are pooled afterwards. It does consume the random stream in a
# different order, so a chunked run is NOT bit-identical to an unchunked one.
# That is why it is off by default and recorded in the fingerprint below.
GEN_CHUNK = int(os.environ.get("QB_GEN_CHUNK", "0"))

# --- RetroDFM-R paper Sec 3.4, "Inference augmentation" -------------------
# The paper builds its candidate list from three multiplied factors:
#   k_a  SMILES augmentation -- re-root the product at different atoms, so the
#        model sees k_a spellings of the same molecule
#   k_s  repeated sampling   -- sample k_s reasoning paths at temperature 1.0
#   k_b  repeated sampling of the ANSWER -- from the <answer> tag of each path,
#        sample k_b reactant strings at temperature 1.4
# giving k_a * k_s * k_b predictions, ranked by Eq. 6:
#   Count(y) = number of times y appears among all k_a*k_s*k_b predictions
#
# THE METHOD CHANGED BETWEEN PAPER VERSIONS. The earlier version used partial
# BEAM SEARCH for the k_b stage (k_b distinct answers, temperature 0) and Eq. 5
# with alpha=1, a rank-decayed score. The current version samples at 1.4 and
# counts plain frequency. This harness followed the old version until
# 2026-09-06; QB_ANS_MODE=beam and QB_ALPHA=1 restore it.
#
# NOT a mistake we made. Both paper versions carry the same paragraph objecting
# to beam search "over the full output space" -- reasoning text plus answer --
# and the OLD version's answer to that objection was partial beam search from
# the <answer> tag, which is what this file did. The authors changed the k_b
# operator; they did not correct an error of ours.
#
# Their OLD inference/eval.sh had SAMPLE_NUM=1, POST_BEAM_SIZE=2, SAMPLE_TEMP=1
# (k_s=1, k_b=2). Their CURRENT eval/eval_two_stage.sh has N_THINK=10,
# N_ANSWER=10, AUGMENTATION=20 -- note the positional order is (k_s, k_b, k_a),
# not (k_a, k_s, k_b), so "10 10 20" IS the paper's (20, 10, 10). k_a never appears in eval.sh: the
# augmented spellings are rows in the released test file, and metric.py merges
# them afterwards by canonical product. We generate them here instead, which is
# equivalent and keeps one record per molecule.
#
# Defaults here are 1/1/1 = one answer per molecule, the setting the current
# paper reports as 60.4% top-1 (Table 3). The full augmented setting
# (20, 10, 10) is reported as 66.1 / 84.2 / 88.6 / 92.2 at top-1/3/5/10.
KA      = int(os.environ.get("QB_KA", "1"))
KS      = int(os.environ.get("QB_KS", "1"))
KB      = int(os.environ.get("QB_KB", "1"))
# Eq. 6 in the paper is Count(y) = sum of I[y_t == y]: a PLAIN FREQUENCY, and
# eval/eval_two_stage.sh passes --alpha 0.0 to score_rsmiles.py. Alpha=1 came
# from an older metric.py and discounts by rank, which is NOT what is published.
ALPHA   = float(os.environ.get("QB_ALPHA", "0.0"))    # paper Eq.6 / their --alpha 0.0
# How the k_a spellings are produced.
#
# "canonical" -- vary the ROOT ATOM, canonical traversal. This is the default
# because it is what the authors actually do. MEASURED against their released
# 50k/reason_aug20.jsonl over 300 products: 36.0% exact set match, mean overlap
# 13.5 of 20, Jaccard 0.734 -- the signature of the same method drawing a
# different random sample.
#
# "random" -- vary root atom AND traversal order. This was a mis-reading of the
# paper's phrase "different starting atoms and molecular graph enumeration".
# Against the same file it scores 0% exact, overlap 3.3, Jaccard 0.117: three
# times WORSE. Kept only so the comparison can be repeated.
AUG     = os.environ.get("QB_AUG", "canonical")
TEMP_S  = float(os.environ.get("QB_TEMP_S", "1.0"))   # eval.sh: SAMPLE_TEMP=1
TOP_P   = float(os.environ.get("QB_TOP_P", "1.0"))    # eval.py: top_p=1
# 0 = top-k filtering DISABLED, which is what the reference implementation does.
#
# VERIFIED on the shipped checkpoint: generation_config.json contains
#     "temperature": 0.6, "top_k": 20, "top_p": 0.95
# and the authors' eval.sh/eval.py use temperature 1, top_p 1, top_k off. This
# harness always passes temperature and top_p, so 0.6 and 0.95 never took
# effect -- but top_k was NOT passed by the earlier runs, so those silently
# inherited 20 from the file. That is the one parameter where this repository
# differed from the reference harness, and it is now closed.
#
# Set QB_TOP_K=20 to use the checkpoint's value instead. Records carry the
# value in their fingerprint, so the two cannot be merged by accident.
#
# 0, NOT -1. The authors run vLLM, where -1 means "disabled". In transformers
# -1 is not a disable value: it is clamped up to min_tokens_to_keep, i.e. 1,
# which silently turns sampling into greedy decoding. Copying their -1 across
# would change the experiment rather than align it.
TOP_K   = int(os.environ.get("QB_TOP_K", "0"))
ANS_MAX = int(os.environ.get("QB_ANS_MAX", "256"))    # eval.sh beam stage
# Generate the k_b answer samples ANS_CHUNK at a time instead of all at once.
# 0 = all at once, the published behaviour. See predict_augmented() for why
# this is the knob that decides whether an 8 GB card can run k_b=10 at all.
ANS_CHUNK = int(os.environ.get("QB_ANS_CHUNK", "0"))
AUGMENTED = (KA * KS * KB) > 1

# --------------------------------------------------------------------------
#  THE SECOND STAGE IS SAMPLING, NOT BEAM SEARCH.  Paper Sec 4.5, verbatim:
#
#    "Direct beam search over the full output space can therefore produce
#     candidates that differ mainly in the intermediate text while converging
#     to identical reactant predictions, limiting diversity at the answer
#     level. To address this issue, we adopt a two-stage inference
#     augmentation strategy that explicitly promotes diversity at the answer
#     generation stage with repeated sampling. ... we generate k_s distinct
#     reasoning trajectories per input. For each trajectory, we then sample
#     k_b reactant predictions by increasing the sampling temperature during
#     answer generation (empirically set to 1.4 in experiments)."
#
#  This harness used deterministic beam search for the k_b stage -- precisely
#  the operator the method was designed to avoid. It also explains the
#  candidate-count anomaly we spent days on: at 20x1x2 only 5.4 distinct
#  candidates came back from 40 predictions, which is the convergence the
#  paragraph above predicts. Beams are distinct by construction, so they carry
#  almost no frequency signal, and Eq. 6 ranks by frequency.
#
#  QB_ANS_MODE=beam restores the old behaviour for a side-by-side comparison.
ANS_MODE   = os.environ.get("QB_ANS_MODE", "sample")        # sample | beam
ANS_TEMP   = float(os.environ.get("QB_ANS_TEMP", "1.4"))    # paper Sec 4.5
THINK_TEMP = float(os.environ.get("QB_THINK_TEMP", "1.1"))  # their eval script

i_sh, n_sh = (int(x) for x in SHARD.split("/"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

rows = [json.loads(l) for l in open(SUBSET) if l.strip()][:N]

# --- Resume, but ONLY from records this exact configuration produced. --------
#
# Two rules fight each other here. Resuming must survive a changed worker count,
# so every shard file for the level is read, not just this worker's own.
# But resuming must NOT survive a changed harness: when the prompt format was
# fixed, 5005 stale records sat on disk, every worker saw its molecules as
# already done, and the run "completed" in three minutes and re-reported the
# old broken numbers as if they were new.
#
# So each record carries a fingerprint of what produced it. A record whose
# fingerprint differs is not evidence that the molecule is done. Bump HARNESS
# whenever prompt construction changes meaning.
HARNESS = "v2-chatml"   # unchanged: the un-augmented arm is unaffected
CFG = (f"{HARNESS}|{LEVEL}|base{BASE_DTYPE}|k{KA}x{KS}x{KB}|"
       f"{'greedy' if GREEDY else f'T{TEMP_S}|p{TOP_P}|k{TOP_K}'}|"
       f"m{MAX_NEW}|s{SEED}"
       # Tag only the NON-default mode. canonical is both the original and the
       # verified-correct behaviour, so records made with it keep the
       # fingerprint they always had and stay resumable; only the experimental
       # mode is marked.
       + (f"|aug{AUG}" if (KA > 1 and AUG != "canonical") else "")
       # Same rule for chunking. Splitting the k_s request draws from the random
       # stream in a different order, so a chunked run is statistically
       # equivalent but not bit-identical -- it must not silently resume a run
       # that was not chunked. Off (0) leaves the fingerprint untouched, so
       # every record made so far stays resumable.
       + (f"|gc{GEN_CHUNK}" if GEN_CHUNK > 0 else "")
       + (f"|ac{ANS_CHUNK}" if ANS_CHUNK > 0 else "")
       # The k_b stage changed from beam search to sampling, and alpha from 1
       # to 0. Records made the old way must never merge with the new ones, but
       # the UN-augmented arm is untouched by both, so it keeps its fingerprint
       # and every 1x1x1 record already on disk stays valid and resumable.
       + (f"|ans{ANS_MODE}T{ANS_TEMP}|th{THINK_TEMP}|a{ALPHA}"
          if AUGMENTED else ""))

import glob as _glob


def count_foreign(path):
    n = 0
    try:
        with open(path) as f:
            for line in f:
                try:
                    if json.loads(line).get("cfg") != CFG:
                        n += 1
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return n


def load_done():
    """Resume from records matching CFG, tolerating fifteen sibling workers
    rearranging the same directory at the same moment.

    Returns (finished, failed): molecules that produced a real answer, and
    molecules whose only records are errors. The second set is work to REDO.
    """
    d = os.path.dirname(OUT) or "."
    prefix = os.path.basename(OUT).rsplit(".s", 1)[0]      # fp16 | fp16.k5x1x2

    # Step 1: settle OUR OWN file before scanning anyone else's, so the wide
    # scan below sees a directory that is done changing on our account.
    if count_foreign(OUT):
        keep = f"{OUT}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            os.rename(OUT, keep)
            print(f"!! records here came from a DIFFERENT configuration.\n"
                  f"   moved to {keep}; regenerating from scratch.", flush=True)
        except FileNotFoundError:
            pass

    # Step 2: scan every shard for this level. A path can disappear between the
    # glob and the open because another worker is renaming its own stale file.
    # That is housekeeping, not an error -- it previously killed the worker with
    # FileNotFoundError before a single molecule had been generated.
    done, failed = set(), set()
    for p in _glob.glob(os.path.join(d, f"{prefix}.s*.jsonl")):
        try:
            fh = open(p)
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue                               # half-written line
                if rec.get("cfg") != CFG:
                    continue
                # A FAILED MOLECULE IS NOT A DONE MOLECULE.
                #
                # Error records carry the right fingerprint, so they used to
                # land in `done` and were never retried. MEASURED on the 4-card
                # nf4 run: 3190 scored + 1358 errored + 457 never attempted =
                # 5005, and every rerun reported "4548 already on disk" and
                # retried none of the 1358. The only way to recover them was to
                # delete records by hand.
                #
                # Excluding them here makes a rerun pick them up. The merge
                # prefers a later success over an earlier failure for the same
                # uid, so retrying costs nothing if it fails again.
                if str(rec.get("stop_reason", "")).startswith("error"):
                    failed.add(rec["uid"])
                    continue
                done.add(rec["uid"])
    return done, failed - done      # a later success cancels an earlier error


# Stagger BEFORE touching the shard directory. All sixteen workers used to run
# load_done() in the same millisecond, which is what made the rename race
# reachable at all; it also spread out the 16 GB of checkpoint reads that
# follow.
time.sleep(min(i_sh * 8, 120))

done, failed = load_done()

# --- DIVIDE WHAT IS LEFT, NOT THE ORIGINAL LIST. ---------------------------
#
# This used to be `rows = rows[i_sh::n_sh]` placed further up, BEFORE anything
# was filtered out: each worker took every n_sh-th molecule of the whole subset
# and only then skipped the finished ones. Correct on a fresh run, wrong on
# every restart with a different number of cards.
#
#   MEASURED. A 4-card run stopped part way, restarted on 2 cards:
#       shard 0/2  ... 0 to do (4548 already on disk)
#       shard 1/2  ... 457 to do (4548 already on disk)
#
# Arithmetic, not luck: old shards 0 and 2 are both even-numbered rows, and
# every even row belongs to new shard 0. Cards 0 and 2 had finished, so shard 0
# owned nothing and one card did all the work. Going the other way (2 cards to
# 4) each old shard fans out over two new ones, which is why THAT direction
# looked fine.
#
# Assign by a hash of the uid rather than by position in the list. Every worker
# then agrees on who owns a molecule without having to agree on the list, so it
# does not matter that they read the directory seconds apart and see slightly
# different sets of finished work -- no molecule can be claimed twice or fall
# through a gap. crc32 and not hash(), which is salted per process.
todo = [r for r in rows
        if r["uid"] not in done
        and zlib.crc32(str(r["uid"]).encode()) % n_sh == i_sh]
n_retry = sum(1 for r in todo if r["uid"] in failed)

# Sidecar so the merge step can filter by the same fingerprint without having
# to reconstruct it in shell (where it would drift out of sync with this file).
with open(OUT + ".cfg", "w") as _f:
    _f.write(CFG + "\n")
print(f"shard {i_sh}/{n_sh}  level {LEVEL}  "
      f"{len(todo)} to do ({len(done)} finished on disk)", flush=True)
if n_retry:
    print(f"   {n_retry} of those {len(todo)} FAILED on an earlier run and are "
          f"being retried", flush=True)

if not todo:
    sys.exit(0)


def prompt(target):
    """Verbatim from the OpenDFM/RetroDFM-R-8B model card; identical to
    src/qb/prompts.py so cluster and laptop runs are comparable."""
    return (f"<SMILES> {target} </SMILES> Given the product SMILES, your task is to "
            f"predict the reactants SMILES using your experienced chemical "
            f"Retrosynthesis knowledge. Please reason step by step, and put your "
            f"final answer within <answer> answer here </answer>.")


def answer_of(text):
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return ""


def canon(s):
    """Eq. 5 aggregates over unique CANONICAL reactants, so votes must be
    counted on the canonical form or the same molecule spelled two ways splits
    its own score."""
    if not s:
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else ""


def roots_of(smi, k, seed):
    """k_a: the same product written k different ways.

    Deterministic given the seed, so a re-run reproduces the same spellings.

    ROOT ATOM *AND* TRAVERSAL ORDER, and the second half was missing.
    RetroDFM-R Sec 3.4: "we select different root atoms and traversal orders to
    obtain k_a distinct representations", following Augmented Transformer [16]
    and R-SMILES [19]. This function used rootedAtAtom with canonical=True,
    which varies the root but PINS the traversal order to the canonical one.
    The number of distinct outputs is then bounded by the molecule's symmetry
    classes rather than by its atom count.

    MEASURED on the first 500 test products, k_a=20 requested:
        canonical=True   17.9 spellings on average, 62% reach 20, minimum 5
        doRandom=True    20.0 spellings on average, 100% reach 20, minimum 18
    An 8-heavy-atom product (NC1(C(F)(F)F)CC1) yields 5 spellings the old way
    and 20 the new way.

    Fewer distinct prompts means fewer distinct reasoning paths, so fewer
    unique candidates survive de-duplication. That barely moves top-1, which
    the consensus still gets right from a handful of spellings, but it starves
    the tail of the ranked list -- which is exactly the shape of the gap
    against the published numbers: top-1/3/5 within a point, top-10 short.

    QB_AUG=canonical restores the old behaviour for comparison. The mode is
    part of the record fingerprint whenever k_a > 1, so the two cannot merge.
    """
    if k <= 1:
        return [smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    out = [Chem.MolToSmiles(mol)]          # canonical spelling always included
    seen = set(out)
    rng = random.Random(seed)
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)

    if AUG == "canonical":
        for r in order:
            if len(out) >= k:
                break
            try:
                s = Chem.MolToSmiles(mol, rootedAtAtom=int(r), canonical=True)
            except Exception:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:k]

    # Root atom AND traversal order, via a seeded permutation of the atom
    # indices: renumbering moves a different atom to position 0 (the root) and
    # reorders the neighbour lists (the traversal), then canonical=False writes
    # the string in that order.
    #
    # NOT doRandom=True, which was the obvious choice and is wrong here: it
    # draws from RDKit's own global RNG, so the same QB_SEED produced different
    # spellings on every call. Measured -- two consecutive calls disagreed.
    # That would silently break both resume and reproducibility of every
    # augmented run.
    #
    # The attempt cap stops a highly symmetric molecule spinning forever when
    # it genuinely cannot produce k distinct strings.
    n_at = mol.GetNumAtoms()
    tries, limit = 0, max(k * 8, 64)
    while len(out) < k and tries < limit:
        tries += 1
        perm = list(range(n_at))
        rng.shuffle(perm)
        try:
            s = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        except Exception:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:k]


def ensure_chat_template():
    """RetroDFM-R is instruction tuned and inference/eval.py wraps every prompt:

        prompt_input(): apply_chat_template(message, add_generation_prompt=True)

    The released checkpoint keeps that template in a SEPARATE chat_template.jinja
    file, not inside tokenizer_config.json. A model directory copied without
    that one file loads perfectly, reports chat_template = None, and then serves
    the model bare prompts in a format it never saw during training -- which is
    silent, and wrong, and is exactly what happened here.

    So: find the template, or stop. Never fall back to the bare string."""
    if getattr(tok, "chat_template", None):
        return "tokenizer"
    # The image's OWN data directory has to be in this list. The build copies
    # chat_template.jinja to /opt/qb/data, and leaving that path out meant an
    # image that contained the file still stopped with "no chat template" and
    # told the user to go and download it.
    cands = [os.environ.get("QB_CHAT_TEMPLATE", ""),
             os.path.join(os.environ.get("QB_ROOT", "/opt/qb"), "data",
                          "chat_template.jinja"),
             "/opt/qb/data/chat_template.jinja",
             os.path.join(MODEL_DIR, "chat_template.jinja"),
             "/work/chat_template.jinja"]
    for cand in cands:
        if cand and os.path.isfile(cand):
            tok.chat_template = open(cand, encoding="utf-8").read()
            return cand
    raise SystemExit(
        "FATAL: no chat template.\n"
        f"  tokenizer at {MODEL_DIR} has none, and none of these exist:\n"
        + "".join(f"    {c or '$QB_CHAT_TEMPLATE (unset)'}\n" for c in cands) +
        "  Get it from huggingface.co/OpenDFM/RetroDFM-R-8B (chat_template.jinja)\n"
        "  and drop it in the work directory. Running without it produces\n"
        "  malformed prompts and meaningless accuracy.")


def encode_prompt(user_text):
    """tokenize=False may already emit special tokens, so do not add them twice."""
    text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")


def beam_prefix_text(user_text, path_text):
    """Reproduces inference/prepare_beam_search.py exactly:

        chat   = apply_chat_template([user, assistant(sampled_path)])
        prompt = chat.rsplit("<answer>", 1)[0] + "<answer>\\n"

    Note rsplit, not split -- the cut is at the LAST tag -- and the trailing
    newline, which is part of the format the beam search continues from."""
    if "<answer>" not in path_text:
        return None
    chat = [{"role": "user", "content": user_text},
            {"role": "assistant", "content": path_text}]
    try:
        full = tok.apply_chat_template(chat, tokenize=False,
                                       add_generation_prompt=False)
    except Exception:
        full = user_text + path_text
    return full.rsplit("<answer>", 1)[0] + "<answer>\n"


def sample_kw():
    if GREEDY:
        return dict(do_sample=False, pad_token_id=tok.eos_token_id)
    # Pass top_k explicitly. Omitting it does NOT disable top-k: transformers
    # falls back to the model's generation_config, which here sets 20 -- so the
    # reported runs used 20 without saying so. Passing it makes the behaviour
    # independent of that file. 0 disables, matching eval.py. temperature and
    # top_p were never affected: they are always passed and so override the
    # config's 0.6 and 0.95.
    return dict(do_sample=True, temperature=TEMP_S, top_p=TOP_P,
                top_k=TOP_K, pad_token_id=tok.eos_token_id)


def _groups(total, size):
    """[total] when size is 0 or larger; otherwise total split into <=size parts."""
    if size <= 0 or size >= total:
        return [total]
    out = []
    while total > 0:
        n = min(size, total)
        out.append(n)
        total -= n
    return out


# Last heartbeat, at module level so the augmentation loop can update it.
_HB = [0.0]


def predict_augmented(target, seed):
    """The paper's two-stage inference. Returns (ranked candidates, tokens)."""
    scores, ntok = {}, 0
    for _ai, smi in enumerate(roots_of(target, KA, seed), 1):    # ---- k_a
        user = prompt(smi)
        ids  = encode_prompt(user)
        plen = ids["input_ids"].shape[1]
        # ---- k_s, in groups of GEN_CHUNK (all at once when GEN_CHUNK is 0).
        # The paths are independent samples pooled by the scorer below, so how
        # many are asked for per call changes peak memory and nothing else.
        paths = []
        for want in _groups(KS, GEN_CHUNK):
            kw = dict(max_new_tokens=MAX_NEW, num_return_sequences=want,
                      **sample_kw())
            # Their eval script runs stage 1 at THINK_TEMPERATURE=1.1, not 1.0.
            if not GREEDY:
                kw["temperature"] = THINK_TEMP
            try:
                # eval.py generates the whole answer here and then throws it
                # away, because the beam stage regenerates it. Stopping at
                # <answer> gives the identical prefix for a fraction of the
                # tokens.
                out_ = net.generate(**ids, stop_strings=["<answer>"],
                                    tokenizer=tok, **kw)
            except TypeError:                   # transformers < 4.39
                out_ = net.generate(**ids, **kw)
            paths.extend(list(out_))
            del out_                            # release this group's KV cache
        for p in paths:
            ntok += max(0, int(p.shape[0]) - plen)
            prefix_txt = beam_prefix_text(user, tok.decode(p[plen:],
                                                           skip_special_tokens=True))
            if prefix_txt is None:
                continue                        # path never reached <answer>
            pids = tok(prefix_txt, return_tensors="pt",
                       add_special_tokens=False).to("cuda")
            blen = pids["input_ids"].shape[1]
            ans_txt = []
            if ANS_MODE == "beam":              # ---- old behaviour, for comparison
                # Beam search is ONE joint search over KB beams, so unlike
                # sampling it cannot be split into independent groups. Left
                # unchunked; ANS_CHUNK does not apply to this branch.
                beams = net.generate(
                    **pids, max_new_tokens=ANS_MAX, do_sample=False,
                    num_beams=KB, num_return_sequences=KB,
                    early_stopping=True, pad_token_id=tok.eos_token_id)
                for b in beams:
                    ntok += max(0, int(b.shape[0]) - blen)
                    ans_txt.append(tok.decode(b[blen:],
                                              skip_special_tokens=True))
                del beams
            else:                               # ---- k_b, the paper's Sec 4.5
                # n=k_b samples at a RAISED temperature. Duplicates are the
                # point: Eq. 6 ranks by how often a reactant set recurs, so a
                # method that returns k distinct strings (beam search) throws
                # the ranking signal away.
                #
                # GREEDY HAS TO WIN HERE. This branch did not exist before the
                # k_b operator changed on 2026-09-06; the old one passed
                # do_sample=False, so QB_GREEDY=1 gave an end-to-end
                # deterministic run -- which is exactly what
                # check_determinism.sh was built on. The new branch sampled at
                # T=1.4 regardless of GREEDY, so two consequences arrived
                # silently: "greedy" runs stopped being reproducible, and an
                # un-augmented 1x1x1 arm was ONE SAMPLE AT T=1.4 rather than
                # the argmax -- which is not the paper's Table 1a baseline and
                # would have scored well below its 59.0% for a reason nothing
                # in the output would have explained.
                #
                # With k_b samples all drawn greedily they would be identical,
                # so ask for one and let Eq.6 rank a single candidate.
                # ---- k_b in groups of ANS_CHUNK (all at once when it is 0).
                # The samples are INDEPENDENT draws at ANS_TEMP pooled by the
                # Eq. 6 frequency count, so how many are asked for per call
                # changes peak memory and nothing else -- the same argument
                # the k_s loop above rests on.
                #
                # WHY THIS EXISTS. num_return_sequences=KB expands one input
                # to a batch of KB and PREFILLS KB copies of the reasoning
                # prefix in a single allocation. That prefix is long (up to
                # MAX_NEW), and the MLP holds three 12288-wide tensors live at
                # once, so at KB=10 it is ~1.3 GB of activation on top of the
                # KV cache. MEASURED 2026-09-26: 8.62 GB reserved against
                # 5.98 GB live, fp4, on an 11 GB card -- which is why the
                # answer stage fails on 8 GB AND 12 GB cards while the k_s
                # stage, which grows one token at a time, never does. It is
                # also why --gen-chunk could not help: that chunks k_s, and
                # every failing traceback pointed at THIS call. Only decoded
                # TEXT crosses between chunks, so one chunk of KV cache is
                # ever resident.
                if GREEDY:
                    _plan, _skw = [1], dict(do_sample=False)
                else:
                    _plan = _groups(KB, ANS_CHUNK)
                    _skw = dict(do_sample=True, temperature=ANS_TEMP,
                                top_p=1.0, top_k=0)
                for _want in _plan:
                    beams = net.generate(
                        **pids, max_new_tokens=ANS_MAX,
                        num_return_sequences=_want,
                        pad_token_id=tok.eos_token_id, **_skw)
                    for b in beams:
                        ntok += max(0, int(b.shape[0]) - blen)
                        ans_txt.append(tok.decode(b[blen:],
                                                  skip_special_tokens=True))
                    del beams               # release this chunk of KV cache
            for rank, txt in enumerate(ans_txt, 1):
                y = canon(txt.split("</answer>", 1)[0].strip())
                if y:                           # ---- metric.py: 1/(1+Alpha*i)
                    scores[y] = scores.get(y, 0.0) + 1.0 / (1.0 + ALPHA * (rank - 1))
            # k_b beams of KV cache stay resident through the NEXT iteration's
            # allocation unless dropped here, so the peak is two beam searches
            # rather than one for no reason.
            del ans_txt, pids
        # HEARTBEAT. The per-molecule progress line only prints once a molecule
        # COMPLETES. At k_a=10 k_s=10 k_b=20 one molecule is 100 reasoning
        # passes plus 100 beam searches -- tens of minutes on one card -- so the
        # log sat silent after "generation flags are not valid" and a perfectly
        # healthy run was indistinguishable from a hung one.
        _now = time.perf_counter()
        if _now - _HB[0] >= 120:
            _HB[0] = _now
            print(f"      ... augmentation {_ai}/{KA}, "
                  f"{len(scores)} distinct candidates so far", flush=True)
    ranked = [y for y, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked, ntok


# bitsandbytes 4-bit levels. These need sm_75 (Turing) or newer, so they run on
# a 2080 Ti and a 3070; torchao's int4 kernels need sm_80.
# Filled in by load(). torchao is a 0.x library: config defaults, including
# the symmetric/asymmetric mapping choice, are not stable across versions.
# Recording the actual field values per run means a reader on a different
# torchao does not have to assume our description applies to them.
QUANT_INFO = {}

BNB = {"nf4":   ("nf4", False),
       "nf4dq": ("nf4", True),
       "fp4":   ("fp4", False)}

# ---------------------------------------------------------------------------
#  WHERE A TORCHAO CONFIG LIVES, AND WHY THAT HAS TO BE SEARCHED
#
#  torchao is a 0.x library and it moves things between releases. Comparing
#  0.9.0 (an older environment) with 0.18.0 (the images used here):
#
#    Int4DynamicActivationInt4WeightConfig  (w4a4)  DELETED. Not present under
#        any name. w4a4 cannot be built on 0.18.0 at all.
#    Int8DynamicActivationInt4WeightConfig  (w4a8)  MOVED out of
#        torchao.quantization into torchao.prototype.quantization.int4, AND
#        CHANGED MEANING: the 0.18.0 class of that name is a CPU backend
#        (Int4OpaqueTensor / da8w4_linear_cpu). Resolving it by name alone would
#        have produced a row labelled "w4a8 on a 5090" that was computed on the
#        CPU. That is worse than a crash, so w4a8 refuses rather than guesses.
#
#  Hence: every level names the module it must come from. A config found in an
#  unexpected module is an error, not a fallback, and the module that supplied
#  it is recorded in meta.json next to the version number.
#
#  Blackwell-only formats (NVFP4, MXFP4, MXFP8) live in torchao.prototype.
#  mx_formats and are gated on compute capability >= (10, 0) -- an RTX 5090 is
#  (12, 0) and clears it, a 3090 is (8, 6) and does not. Whether the KERNELS
#  were compiled for sm_120 in this particular wheel is a separate question,
#  which is what probe_capability.py exists to answer.
# ---------------------------------------------------------------------------
_AOQ = "torchao.quantization"
_MX  = "torchao.prototype.mx_formats"


def _fp4():
    return torch.float4_e2m1fn_x2


def _fp8():
    return torch.float8_e4m3fn


def _int2():
    # torch.int1..int7 are sub-byte dtypes added in torch 2.6. On an older
    # torch this raises AttributeError, which _build now catches so the next
    # candidate spelling (uint2, for torchao 0.9.0) gets its turn.
    return torch.int2


def _uint2():
    return torch.uint2


def _per_group(n):
    from torchao.quantization import PerGroup
    return PerGroup(n)


# ---------------------------------------------------------------------------
#  TWO-BIT: WHAT THIS MEASURES, AND WHAT IT DOES NOT
#
#  int2_ao and w2a8_intx quantize weights to 2 bits and are faithful for
#  ACCURACY: the weights really do carry 4 distinct values per group, and the
#  top-k they produce is the top-k of a 2-bit model.
#
#  They are NOT a memory or speed result. torchao's portable intx path is
#  IntxPackingFormat.UNPACKED_TO_INT8 -- one 2-bit value stored per int8 byte,
#  dequantized to bf16 for the matmul. So a 2-bit arm occupies MORE memory than
#  int8_ao, not less, and runs no faster. The packed alternatives in torchao
#  0.18 are all OPAQUE_* (KleidiAI / lowbit), which are ARM-CPU backends -- the
#  same class of trap as w4a8 above, where a CUDA-labelled row would have been
#  computed on the CPU.
#
#  This is already true of w4a8_intx and was measured: 9.12 GB for "4-bit"
#  against 8.22 GB for int8.
#
#  So: quote 2-bit in the accuracy-versus-bits curve. Do NOT put it in the
#  hardware-feasibility table, and do not write that 2 bits halves 4-bit's
#  footprint. For a real packed 2-bit footprint the options are GGUF q2_k
#  (needs llama.cpp, absent from image B) or HQQ (not installed) -- both a new
#  image, which is why they are not here.
# ---------------------------------------------------------------------------


# level -> list of (module, attribute, kwargs-factory). Tried in order.
AO_SPECS = {
    # --- integer levels, unchanged from the published runs -----------------
    "int8_ao": [(_AOQ, "Int8WeightOnlyConfig", dict),
                (_AOQ, "int8_weight_only", dict)],
    # int4_ao asks for the TINYGEMM packing explicitly. torchao 0.18 changed the
    # DEFAULT packing to "plain", whose from_hp() imports the separate mslk
    # package and raises "ImportError: Requires mslk >= 1.0.0" -- MEASURED on an
    # RTX 3070 with torchao 0.18.0+cu130, and it would fail identically on any
    # other card because it is a packaging gap, not a hardware limit.
    # tile_packed_to_4d routes to torch.ops.aten._weight_int4pack_mm, which is
    # the same tinygemm kernel torchao 0.9.0 used for the published int4_ao
    # numbers -- so this restores the level AND keeps it comparable.
    # The kwarg is dropped automatically on 0.9.0, which has no such field.
    "int4_ao": [(_AOQ, "Int4WeightOnlyConfig",
                 lambda: {"int4_packing_format": "tile_packed_to_4d"}),
                (_AOQ, "Int4WeightOnlyConfig", dict),
                (_AOQ, "int4_weight_only", dict)],
    "w8a8":    [(_AOQ, "Int8DynamicActivationInt8WeightConfig", dict),
                (_AOQ, "int8_dynamic_activation_int8_weight", dict)],
    "w4a8":    [(_AOQ, "Int8DynamicActivationInt4WeightConfig", dict),
                (_AOQ, "int8_dynamic_activation_int4_weight", dict)],
    "w4a4":    [(_AOQ, "Int4DynamicActivationInt4WeightConfig", dict),
                (_AOQ, "int4_dynamic_activation_int4_weight", dict)],
    # w4a8 rebuilt from the general intx config, for torchao >= 0.14 where the
    # dedicated class is gone. This is NOT an alias for w4a8 and must not be
    # reported as one -- different kernel, so it gets its own level name and
    # its own row in the table.
    #
    # The RECIPE is identical, which is why it is worth having: torchao 0.9.0's
    # Int8DynamicActivationInt4WeightConfig was group_size=32, SYMMETRIC
    # weights, ASYMMETRIC activations, and this config's defaults are
    # PerGroup(32), MappingType.SYMMETRIC, MappingType.ASYMMETRIC -- the same
    # three choices, field for field.
    #
    # Default packing is UNPACKED_TO_INT8: one int4 value per byte. That also
    # explains an earlier measurement where w4a8 held 9.12 GB
    # against int8's 8.22 GB -- 4-bit weights that occupy 8 bits each cost more
    # than 8-bit weights plus their scales.
    "w4a8_intx": [(_AOQ, "Int8DynamicActivationIntxWeightConfig",
                   lambda: {"weight_dtype": torch.int4})],
    # --- 8-bit float, sm_89+ (Ada) and Blackwell ---------------------------
    "fp8_ao":  [(_AOQ, "Float8WeightOnlyConfig", dict)],
    "fp8_w8a8": [(_AOQ, "Float8DynamicActivationFloat8WeightConfig", dict)],
    "fp8_w4a8": [(_AOQ, "Float8DynamicActivationInt4WeightConfig", dict)],
    # --- 4-bit float, Blackwell only ---------------------------------------
    "nvfp4":   [(_MX, "NVFP4DynamicActivationNVFP4WeightConfig", dict)],
    "nvfp4_w": [(_MX, "NVFP4WeightOnlyConfig", dict)],
    "mxfp8":   [(_MX, "MXDynamicActivationMXWeightConfig",
                 lambda: {"activation_dtype": _fp8(), "weight_dtype": _fp8()})],
    "mxfp4":   [(_MX, "MXDynamicActivationMXWeightConfig",
                 lambda: {"activation_dtype": _fp4(), "weight_dtype": _fp4()})],
    # --- 2-bit, added 2026-09-14 -------------------------------------------
    # Two spellings because two torchao releases have non-overlapping APIs.
    # VERIFIED by reading both wheels, not inferred:
    #   0.18.0 (the image's release): IntxWeightOnlyConfig asserts weight_dtype
    #       in [torch.int1 .. torch.int8], so int2 is explicitly in range.
    #       UIntXWeightOnlyConfig is GONE from this release.
    #   0.9.0  (older release): no IntxWeightOnlyConfig at all;
    #       UIntXWeightOnlyConfig(dtype=torch.uint2, group_size=...) is the
    #       path, and uintx_layout maps torch.uint2 -> 2 bits.
    # Neither spelling alone covers both releases.
    #
    # group 32 matches w4a8_intx, so the 4-bit and 2-bit rows differ in bit
    # width and nothing else.
    "int2_ao": [(_AOQ, "IntxWeightOnlyConfig",
                 lambda: {"weight_dtype": _int2(),
                          "granularity": _per_group(32)}),
                (_AOQ, "UIntXWeightOnlyConfig",
                 lambda: {"dtype": _uint2(), "group_size": 32}),
                (_AOQ, "uintx_weight_only",
                 lambda: {"dtype": _uint2(), "group_size": 32})],
    # 2-bit weights with int8 dynamic activations -- the 2-bit sibling of
    # w4a8_intx, same class, same defaults, weight_dtype the only change.
    "w2a8_intx": [(_AOQ, "Int8DynamicActivationIntxWeightConfig",
                   lambda: {"weight_dtype": _int2()})],
}

# MX asserts weight.dtype == bfloat16, so these levels cannot run on an fp16
# base. Checked explicitly rather than left to an assertion deep in torchao.
MX_LEVELS = {"mxfp8", "mxfp4"}

# Levels whose kwargs ARE the measurement: they carry the bit width or the
# element format, and every one of these torchao configs has a default that is
# something else. _build must not fall back to a bare constructor for them --
# see the comment at the fallback itself.
STRICT_KWARGS = {"int2_ao", "w2a8_intx", "w4a8_intx", "mxfp8", "mxfp4"}

AO_LEVELS = set(AO_SPECS)


def resolve_ao_config(level):
    """Build the torchao config for a level. Returns (config, module_it_came_from).

    Raises with the full list of what was tried rather than a bare AttributeError,
    because "this level is not in this build" is a result worth reporting and the
    three ways it can happen look identical from the outside otherwise.
    """
    import importlib
    tried = []
    for modname, attr, kw in AO_SPECS[level]:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            tried.append(f"{modname}: import failed ({type(e).__name__})")
            continue
        obj = getattr(mod, attr, None)
        if obj is None:
            tried.append(f"{modname}.{attr}: absent")
            continue
        try:
            kwargs = kw()
        except Exception as e:
            # The FACTORY can fail before the config is ever built: torch.int2
            # does not exist below torch 2.6, and PerGroup is not importable
            # from torchao.quantization on 0.9.0. This used to escape _build as
            # a bare AttributeError instead of falling through to the next
            # candidate spelling, which defeats the entire point of the list.
            tried.append(f"{modname}.{attr}: kwargs unavailable "
                         f"({type(e).__name__}: {e})")
            continue
        try:
            cfg = obj(**kwargs)
        except TypeError:
            # An older torchao without these fields. Retry bare rather than
            # declaring the level unavailable: the kwargs are refinements, not
            # requirements, and 0.9.0 must keep resolving exactly as it did.
            #
            # EXCEPT where the kwargs ARE the measurement. Bare-retry an int2
            # request and IntxWeightOnlyConfig hands back its default,
            # weight_dtype=torch.int8 -- an 8-bit model in a row labelled
            # 2-bit, with nothing in the log to show for it. Same for int4 and
            # for the MX element formats. A level that will not build with its
            # own kwargs is unavailable; it is not an invitation to substitute.
            if level in STRICT_KWARGS:
                tried.append(f"{modname}.{attr}: rejected {kwargs}, and a bare "
                             f"retry would silently change the bit width")
                continue
            try:
                cfg = obj()
                kwargs = {}
            except Exception as e:
                tried.append(f"{modname}.{attr}: construction failed "
                             f"({type(e).__name__}: {e})")
                continue
        except Exception as e:
            tried.append(f"{modname}.{attr}: construction failed "
                         f"({type(e).__name__}: {e})")
            continue
        suffix = f" {kwargs}" if kwargs else ""
        return cfg, f"{modname}.{attr}{suffix}"

    msg = [f"level '{level}' is not available in torchao {_ao_version()}."]
    for t in tried:
        msg.append(f"    tried {t}")
    if level == "w4a4":
        msg.append("    Int4DynamicActivationInt4WeightConfig was removed after "
                   "torchao 0.9.x. Use torchao 0.9.0 for w4a4, or nvfp4 for "
                   "4-bit activations on Blackwell.")
    if level in ("int2_ao", "w2a8_intx"):
        msg.append("    2-bit needs either torchao >= 0.14 (IntxWeightOnlyConfig, "
                   "weight_dtype=torch.int2, which also needs torch >= 2.6 for "
                   "the sub-byte dtypes) or torchao <= 0.13 "
                   "(UIntXWeightOnlyConfig, dtype=torch.uint2). Run "
                   "probe_capability.py on this node to see which, if "
                   "either, this image has.")
    if level == "w4a8":
        msg.append("    NOTE: torchao >= 0.17 ships a class of this name under "
                   "torchao.prototype.quantization.int4, but it is a CPU "
                   "backend (da8w4_linear_cpu), not the CUDA path measured on "
                   "0.9.0. It is deliberately NOT used here: it would report "
                   "CPU numbers under a GPU label.")
        msg.append("    Use level 'w4a8_intx' instead: same recipe (group 32, "
                   "symmetric weights, asymmetric activations) built from "
                   "Int8DynamicActivationIntxWeightConfig, different kernel.")
    raise RuntimeError("\n".join(msg))


def _ao_version():
    try:
        import torchao
        return torchao.__version__
    except Exception:
        return "?"


def gpu_total_gb():
    """Usable VRAM in GB, honouring QB_MEM_LIMIT_GB.

    QB_MEM_LIMIT_GB=8 makes the worker behave as though the card had 8 GB:
    the loading-path branch below sees 8, and the allocator is capped at 8 so
    an over-large model actually fails instead of quietly succeeding. That is
    the point -- it answers "would this fit on a smaller card?" on hardware you
    already have, without borrowing one."""
    lim = float(os.environ.get("QB_MEM_LIMIT_GB", "0"))
    real = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(lim, real) if lim > 0 else real


# ---------------------------------------------------------------------------
#  SPILLING TO HOST RAM INSTEAD OF FAILING
#
#  A CUDA allocation does not overflow into system memory by itself. On Linux
#  it either fits in VRAM or raises OutOfMemoryError -- there is no automatic
#  fallback. (The Windows/WSL driver does spill silently since 536.40, which is
#  a different platform and a measurement hazard rather than a feature.)
#
#  The supported way to use host RAM deliberately is accelerate's placement
#  planner: give it a per-device budget and it puts as many whole layers on the
#  GPU as fit, leaves the rest in host RAM, and streams each one across the bus
#  when its turn comes. QB_OFFLOAD=1 turns that on, using QB_MEM_LIMIT_GB as
#  the GPU budget.
#
#  This changes what is being measured, so it is NOT the default and offloaded
#  runs are written to their own files. Streaming layers over PCIe costs far
#  more than the arithmetic saved, so seconds-per-molecule from an offloaded
#  run is not comparable with a fully-resident one. What it answers is a
#  different and still useful question: not "does this fit", but "what does it
#  cost when it does not".
# ---------------------------------------------------------------------------
OFFLOAD = os.environ.get("QB_OFFLOAD", "0") == "1"

# ---------------------------------------------------------------------------
#  OFFLOAD AND torchao QUANTIZATION DO NOT MIX.
#
#  MEASURED 2026-09-16, RTX 3070 Laptop, int2_ao with QB_OFFLOAD=1:
#
#      File ".../torchao/quantization/quant_primitives.py", line 215, in forward
#        return torch.round(x)
#      RuntimeError: CUDA driver error: device not ready
#
#  quantize_() walks the modules and rewrites each weight in place, but under
#  offload the weights are on meta/CPU behind accelerate's hooks and are moved
#  only when a forward pass reaches them. Quantizing something that is not
#  resident gives a driver error from deep inside a rounding kernel, which
#  names neither offload nor quantization.
#
#  bitsandbytes levels are fine: they quantize during from_pretrained, before
#  any offload hooks exist.
#
#  This refuses rather than warns because the failure costs a model load to
#  discover and its message points nowhere near the cause. QB_OFFLOAD_FORCE=1
#  overrides if you want to see it for yourself.
# ---------------------------------------------------------------------------
if OFFLOAD and os.environ.get("QB_OFFLOAD_FORCE", "0") != "1":
    _ao_lv = LEVEL not in ("fp16", "bf16") and LEVEL not in ("nf4", "nf4dq", "fp4")
    if _ao_lv:
        sys.exit(
            f"QB_OFFLOAD=1 cannot be combined with the torchao level '{LEVEL}'.\n"
            "  quantize_() rewrites weights in place; under offload they are on\n"
            "  meta/CPU behind accelerate hooks and are not resident yet, which\n"
            "  fails as 'CUDA driver error: device not ready' inside a rounding\n"
            "  kernel -- a message that names neither cause.\n"
            "  On a card too small for the level, use a bitsandbytes level\n"
            "  instead: nf4 measures 0.516 bytes/weight and really does pack,\n"
            "  against 1.094 for every torchao intx level including int2_ao.\n"
            "  Set QB_OFFLOAD_FORCE=1 to override.")


def placement_kwargs():
    """from_pretrained kwargs deciding where the weights live."""
    if not OFFLOAD:
        return {"device_map": {"": 0}}
    budget = float(os.environ.get("QB_MEM_LIMIT_GB", "0")) or gpu_total_gb()
    # accelerate plans from max_memory, but activations and the KV cache have
    # to fit ALONGSIDE the weights it places. Handing it the whole budget fills
    # the card with layers and then dies in the first forward pass, which looks
    # like the offload not working. Keep a margin back.
    head = float(os.environ.get("QB_OFFLOAD_HEADROOM", "0.85"))
    gpu_gb = max(1.0, budget * head)
    cpu_gb = os.environ.get("QB_OFFLOAD_CPU_GB", "120")
    print(f"CPU offload ON: up to {gpu_gb:.1f} GiB on the GPU "
          f"(of a {budget:.1f} GB budget, {head:.0%} of it), "
          f"remainder in host RAM (limit {cpu_gb} GiB)", flush=True)
    return {"device_map": "auto",
            "max_memory": {0: f"{gpu_gb:.1f}GiB", "cpu": f"{cpu_gb}GiB"}}


def report_placement(net):
    """How much actually ended up off the card -- the number worth reporting."""
    dm = getattr(net, "hf_device_map", None)
    if not dm:
        return None
    on_cpu = sorted(k for k, v in dm.items() if str(v) in ("cpu", "disk"))
    if not on_cpu:
        # This line used to say "offload requested but not needed" ALWAYS,
        # because it never checked OFFLOAD. Without --offload the device_map is
        # {"": 0} -- a single entry meaning "the whole model on GPU 0" -- so it
        # printed "all 1 modules on the GPU (offload requested but not needed)"
        # on every ordinary run and read like a warning about something the
        # user had not asked for. Neither half was true.
        if OFFLOAD:
            print(f"placement: all {len(dm)} entries on the GPU -- offload was "
                  f"requested but nothing had to be moved", flush=True)
        else:
            print(f"placement: whole model on GPU 0 (no offload)", flush=True)
    else:
        print(f"placement: {len(on_cpu)} of {len(dm)} modules in host RAM, "
              f"{len(dm) - len(on_cpu)} on the GPU", flush=True)
        print(f"  first offloaded: {', '.join(on_cpu[:4])}"
              f"{' ...' if len(on_cpu) > 4 else ''}", flush=True)
    return {"n_modules": len(dm), "n_offloaded": len(on_cpu)}


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    if LEVEL in BNB:
        # bitsandbytes quantizes DURING loading, shard by shard, so the full
        # 16 GB bf16 model never exists anywhere. Peak GPU is roughly the final
        # quantized size. That is what makes an 8B model fit an 11 GB card.
        from transformers import BitsAndBytesConfig
        qt, dq = BNB[LEVEL]
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qt,
            bnb_4bit_use_double_quant=dq,
            bnb_4bit_compute_dtype=(torch.bfloat16 if BASE_DTYPE == "bf16"
                                    else torch.float16))
        net = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb, low_cpu_mem_usage=True,
            **placement_kwargs())
        return tok, net.eval(), f"bitsandbytes {qt}{' +double-quant' if dq else ''}"
    if LEVEL in ("fp16", "bf16"):
        dt = torch.float16 if LEVEL == "fp16" else torch.bfloat16
        # device_map streams each shard STRAIGHT TO THE GPU. Without it,
        # from_pretrained(...).cuda() assembles the whole 16 GB model in host
        # RAM first; sixteen workers doing that at once asked for 256 GB and the
        # OOM killer took every one of them -- SIGKILL, so the logs just stopped
        # mid-load with no traceback. With device_map, host RAM per worker is
        # about one shard.
        net = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=dt, low_cpu_mem_usage=True,
            **placement_kwargs())
        return tok, net.eval(), f"{LEVEL} weights (device_map)"
    import torchao
    import torchao.quantization as Q
    # BASE DTYPE -- the dtype the quantized weights are dequantized into and
    # the matmuls accumulate in. It is not cosmetic.
    #
    # OBSERVED, in two stages. A full int8_ao run recorded 5005 records,
    # 5005 "error:RuntimeError" and 5005 empty prediction lists at 0.231
    # s/molecule -- it failed instantly on every molecule. That run recorded
    # only the exception CLASS, so a 16-molecule rerun was used to capture the
    # message: "probability tensor contains either `inf`, `nan` or element
    # < 0" out of torch.multinomial. Note the failing run's meta.json predates
    # the base_dtype field, so fp16 is inferred from it having been the
    # default at the time, not read off the record.
    #
    # CONJECTURED, NOT MEASURED: fp16 tops out at 65,504, this checkpoint is a
    # Qwen-family model trained in bf16 which has fp32's range, so an
    # intermediate plausibly overflows once quantization rescales the matmuls.
    # No activation magnitude has ever been recorded to confirm this, so it
    # should not be repeated anywhere it will be read as fact. w8a8 was unaffected, which is consistent with
    # the conjecture -- dynamic activation quantization rescales per tensor --
    # but is not evidence for it.
    #
    # bf16 is therefore the default base for every torchao level. Override with
    # QB_BASE_DTYPE=fp16 to reproduce the overflow deliberately -- it is a
    # result worth reporting, not just a bug to hide.
    base = {"fp16": torch.float16, "bf16": torch.bfloat16}[BASE_DTYPE]
    # device_map here too. The laptop version loaded to host RAM and quantized
    # with device="cuda", which is fine for ONE process; sixteen of them each
    # assembling 16 GB in host RAM is the 256 GB request that got every worker
    # SIGKILLed. Load straight onto the card, then quantize in place on the GPU:
    # 16 GB of fp16 weights on a 32 GB card leaves ample room to convert.
    cfg, where = resolve_ao_config(LEVEL)
    if LEVEL in MX_LEVELS and base is not torch.bfloat16:
        raise RuntimeError(
            f"{LEVEL} requires QB_BASE_DTYPE=bf16; torchao asserts "
            f"weight.dtype == bfloat16 for MX formats, got {BASE_DTYPE}")
    try:
        import dataclasses
        QUANT_INFO.update({
            "torchao": torchao.__version__,
            "config": type(cfg).__name__,
            "config_module": where,
            "fields": {f.name: str(getattr(cfg, f.name, None))
                       for f in dataclasses.fields(cfg)},
        })
    except Exception as _e:
        QUANT_INFO.update({"config": type(cfg).__name__,
                           "config_module": where,
                           "fields_error": f"{type(_e).__name__}: {_e}"})

    # QB_FORCE_SMALL_CARD=1 takes the host-RAM path even on a large card.
    # The big-card path below uses device_map, which under some
    # torch/accelerate combinations leaves parameters on the meta device
    # and fails with "Cannot copy out of meta tensor". The staged path
    # loads to CPU first, which is verified to work, at the cost of
    # ~16 GB of host RAM and a slower load.
    if gpu_total_gb() >= 20 and os.environ.get("QB_FORCE_SMALL_CARD", "0") != "1":
        # Big card: hold the whole base model on the GPU, then convert. Fastest,
        # and host RAM stays at about one checkpoint shard -- which is what
        # sixteen concurrent workers need.
        net = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=base, low_cpu_mem_usage=True,
            **placement_kwargs())
        Q.quantize_(net, cfg)
        how = "device_map"
    else:
        # Small card (11 GB 2080 Ti, 8 GB 3070): 16 GB of base weights will not
        # fit, so stage them in HOST RAM and let torchao convert and move one
        # module at a time. Peak GPU is the quantized model plus one module.
        # Costs ~16 GB of system RAM, so run ONE worker per machine.
        net = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=base, low_cpu_mem_usage=True)
        try:
            Q.quantize_(net, cfg, device="cuda")
        except TypeError:                    # older torchao: no device kwarg
            Q.quantize_(net, cfg)
            net = net.cuda()
        how = "streamed from host RAM"
    gc.collect(); torch.cuda.empty_cache()      # drop the freed base tensors
    return tok, net.eval(), f"torchao {where} ({how})"


_lim = float(os.environ.get("QB_MEM_LIMIT_GB", "0"))

# WHICH PHASE THE CAP APPLIES TO -- and this distinction is the whole point.
#
# A quantized model is built by materialising higher-precision weights and
# converting them, so the PEAK while loading is far above what stays RESIDENT
# afterwards. Capping both answers "could this card build AND run the model";
# capping only inference answers "could this card RUN a model quantized
# elsewhere". They are different claims and the second is the one a deployment
# usually cares about, because conversion is a one-off you can do on any
# machine and ship.
#
# MEASURED: int4_ao with QB_MEM_LIMIT_GB=8 dies inside
# _quantize_affine_tinygemm with ~7 GB already resident and a 1.16 GB temporary
# still to allocate -- so its conversion peak exceeds 8 GB even though its
# resident weights are around 5 GB. bitsandbytes nf4 has no such peak because
# it quantizes shard by shard during loading, which is exactly why nf4 fits an
# 8 GB card and int4_ao does not.
#
#     all   (default) cap everything, loading included
#     infer cap only after the model is resident
_PHASE = os.environ.get("QB_MEM_LIMIT_PHASE", "all").lower()


def _apply_mem_cap(when):
    if _lim <= 0 or when != _PHASE:
        return
    _real = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, _lim / _real), 0)
    print(f"VRAM limited to {_lim:.1f} GB of {_real:.1f} GB "
          f"(simulating a smaller card; phase={_PHASE})", flush=True)


# ---------------------------------------------------------------------------
#  WHAT THE DRIVER SEES, which is what decides whether a card is big enough.
#
#  Every number above comes from torch's caching allocator. By construction
#  memory_reserved() >= memory_allocated(): reserved is the pool torch has taken
#  from the driver, allocated is the live tensors inside it. Neither, however,
#  counts what the PROCESS holds outside that pool -- the CUDA context itself
#  (typically 0.3-0.6 GB), cuBLAS/cuDNN workspaces, and any library that calls
#  cudaMalloc directly. So max_memory_reserved() UNDERSTATES the card size this
#  job needs, which is why a run that reports 7 GB can still fail on an 8 GB
#  card.
#
#  torch.cuda.mem_get_info() asks the driver instead: (free, total) for the
#  device. total - free is everything resident on the card at that instant. A
#  daemon thread samples it every QB_MEM_POLL ms and keeps the maximum, so the
#  peak is caught between progress lines rather than only at them.
#
#  CAVEAT, and it belongs next to the number: on a shared card total - free also
#  counts other processes. One worker per GPU (as here) makes it this job's
#  footprint; anything else makes it an upper bound.
# ---------------------------------------------------------------------------
_MEM_POLL_MS = int(os.environ.get("QB_MEM_POLL", "200"))
_DRIVER_PEAK = [0.0]
_DEV_TOTAL = [0.0]


def _driver_used_gb():
    free, total = torch.cuda.mem_get_info()
    _DEV_TOTAL[0] = total / 1024**3
    return (total - free) / 1024**3


def _start_mem_sampler():
    if _MEM_POLL_MS <= 0:
        return
    import threading

    def _loop():
        while True:
            try:
                u = _driver_used_gb()
                if u > _DRIVER_PEAK[0]:
                    _DRIVER_PEAK[0] = u
            except Exception:
                return          # device gone; stop quietly, never kill the run
            time.sleep(_MEM_POLL_MS / 1000.0)

    threading.Thread(target=_loop, daemon=True).start()


_apply_mem_cap("all")
try:
    _driver_used_gb()
    _start_mem_sampler()
    print(f"driver memory sampler on, every {_MEM_POLL_MS} ms "
          f"(card total {_DEV_TOTAL[0]:.2f} GB)", flush=True)
except Exception as _e:
    print(f"driver memory sampler unavailable: {type(_e).__name__}: {_e}",
          flush=True)

torch.manual_seed(SEED)
t0 = time.perf_counter()
try:
    tok, net, detail = load()
except torch.cuda.OutOfMemoryError as e:
    # A refusal is a result. Report it as one rather than as a traceback: the
    # simulated cap working correctly looks identical to the option being
    # broken unless the difference is spelled out.
    _real = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("\n" + "=" * 72, flush=True)
    print(f"RESULT: {LEVEL} does NOT fit in {_lim:.1f} GB." if _lim > 0 else
          f"RESULT: {LEVEL} does not fit on this {_real:.1f} GB card.", flush=True)
    print("=" * 72, flush=True)
    if _lim > 0:
        print(f"  The card really has {_real:.1f} GB; the run was capped at "
              f"{_lim:.1f} GB on purpose (--mem-gb).", flush=True)
        print("  The cap worked -- this is the answer, not a crash.", flush=True)
    print(f"  It ran out during LOADING, not generation. For a quantized level "
          f"the\n  conversion peak is well above the resident size, so this does "
          f"not mean\n  the model could not RUN in {_lim:.1f} GB.", flush=True)
    print("\n  To separate the two questions:", flush=True)
    print("    QB_MEM_LIMIT_PHASE=infer   cap inference only, load uncapped",
          flush=True)
    print("    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   less "
          "fragmentation", flush=True)
    print(f"\n  torch said: {str(e).splitlines()[0]}", flush=True)
    raise SystemExit(8)
tmpl_src = ensure_chat_template()
print(f"chat template from: {tmpl_src}", flush=True)

# NUMERICAL SANITY -- one forward pass before committing to 5005 molecules.
# int8_ao once produced nan logits on every molecule and the run still took
# hours across sixteen GPUs to tell us. This costs a fraction of a second.
with torch.no_grad():
    _p = encode_prompt(prompt("CCO"))
    _lg = net(**_p).logits
_finite = bool(torch.isfinite(_lg).all())
print(f"logit sanity: finite={_finite} "
      f"absmax={_lg.abs().max().item():.4g} dtype={_lg.dtype}", flush=True)
if not _finite:
    raise SystemExit(
        f"FATAL: {LEVEL} on a {BASE_DTYPE} base produces non-finite logits.\n"
        "  Sampling from these raises 'probability tensor contains inf/nan'.\n"
        "  If base is fp16, retry with QB_BASE_DTYPE=bf16: this checkpoint is\n"
        "  natively bf16 and fp16 overflows at 65504 once weights are quantized.")
del _p, _lg
gc.collect(); torch.cuda.empty_cache()
load_s = round(time.perf_counter() - t0, 1)
# Two different numbers, and conflating them would misreport the whole paper.
# A quantized model is built by loading fp16 first and converting, so the PEAK
# during loading is ~16 GB whatever the target precision. What shrinks is what
# stays RESIDENT afterwards. Measure resident here, then reset the peak counter
# so the generation peak below is attributable to inference alone.
resident = round(torch.cuda.memory_allocated() / 1024**3, 2)
load_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
torch.cuda.reset_peak_memory_stats()
print(f"loaded in {load_s}s | {detail} | {resident} GB resident "
      f"({load_peak} GB peak while loading)", flush=True)
PLACEMENT = report_placement(net)
if PLACEMENT and PLACEMENT["n_offloaded"]:
    print("  NOTE: seconds/molecule from an offloaded run measures PCIe "
          "traffic,\n  not arithmetic, and must not be compared with a "
          "fully-resident run.", flush=True)

# With phase=infer the cap goes on HERE, once the weights are resident, so the
# question being asked is "can this card run it", not "can this card build it".
_apply_mem_cap("infer")

n_done, n_err, tok_total, t_run = 0, 0, 0, time.perf_counter()
t_last = t_run
with open(OUT, "a") as out:
    for i, r in enumerate(todo, 1):
        t1 = time.perf_counter()
        try:
            torch.manual_seed(SEED + i)      # per-molecule, so a re-run repeats
            if AUGMENTED:
                with torch.no_grad():
                    cands, ntok = predict_augmented(r["target"], SEED + i)
                stop = "eos" if cands else "empty"
            else:
                # One sampled answer -- the paper's "single-generation" setting.
                # TARGET NUMBER: 60.4% top-1 (2026 version, Table 3, Qwen3-8B
                # checkpoint -- the one we run). NOT the 59.0% of the earlier
                # version, which was the ChemDFM-v1.5 model and a different
                # network entirely. Our measurement is 59.6% [58.2, 60.9] at
                # n=5005. The paper does not state the decoding parameters for
                # this setting, so the comparison is approximate.
                # Sampling parameters are eval.py's, NOT the model card's
                # 0.6/0.9/20 that this script used until the reference harness
                # was read -- those correspond to no published number.
                ids = encode_prompt(prompt(r["target"]))
                with torch.no_grad():
                    gen = net.generate(**ids, max_new_tokens=MAX_NEW, **sample_kw())
                new = gen[0][ids["input_ids"].shape[1]:]
                txt = tok.decode(new, skip_special_tokens=True)
                ans = answer_of(txt)
                cands = [ans] if ans else []
                stop = "eos" if len(new) < MAX_NEW else "length"
                ntok = int(len(new))
        except torch.cuda.OutOfMemoryError:
            cands, stop, ntok = [], "error:OutOfMemoryError", 0
            torch.cuda.empty_cache()
            # Count it. This branch used to leave n_err alone, so a run losing a
            # quarter of its molecules to memory still printed "[0 errors]" on
            # every progress line and looked healthy until scoring, hours later.
            n_err += 1
        except Exception as e:
            # Record the MESSAGE, not just the class. 5005 molecules once failed
            # with "error:RuntimeError" and nothing else, which said only that
            # something broke -- not what, and not where.
            n_err += 1
            if n_err == 1:
                print("FIRST ERROR -- full traceback:", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
            cands, ntok = [], 0
            stop = ("error:" + f"{type(e).__name__}: {e}".replace("\n", " ")[:300])

            # IS THE CUDA CONTEXT STILL ALIVE?
            #
            # A device-side assert (out-of-range index, or multinomial handed a
            # NaN/Inf probability) does not just fail one call -- it poisons the
            # whole context, and EVERY later CUDA call in this process fails
            # too. Catching per molecule and continuing then turns one fault
            # into a hundred, and the run burns hours writing error records.
            #
            # MEASURED: a 20-molecule run failed at molecule 7 and then failed
            # every remaining molecule, 14 errors in 8.4 hours, output useless.
            #
            # So probe the context with a trivial op. If it is dead, stop now
            # and say so; the molecules already written stay valid and a rerun
            # resumes from them.
            try:
                torch.zeros(1, device="cuda").add_(1).item()
            except Exception as probe_err:
                print("\n" + "=" * 72, flush=True)
                print("CUDA CONTEXT IS DEAD -- stopping this shard.", flush=True)
                print("=" * 72, flush=True)
                print(f"  triggered by  : {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)
                print(f"  probe also failed: {type(probe_err).__name__}", flush=True)
                print(f"  molecules completed before this: {n_done}", flush=True)
                print("", flush=True)
                # The card's own counters, read at the moment of the fault.
                # nvidia-smi needs no privileges, and ECC/remapped-row state is
                # the one hardware readout available when the kernel log is not
                # (SLURM batch jobs, no sudo, journal locked down). Captured
                # here because minutes later the node may be running something
                # else -- or have rebooted.
                try:
                    import subprocess
                    print("-" * 72, flush=True)
                    print("CARD COUNTERS AT THE MOMENT OF THE FAULT", flush=True)
                    print("-" * 72, flush=True)
                    out = subprocess.run(
                        ["nvidia-smi", "-q", "-d", "ECC,ROW_REMAPPER,PAGE_RETIREMENT,"
                         "TEMPERATURE,POWER,CLOCK,PERFORMANCE"],
                        capture_output=True, text=True, timeout=60).stdout
                    print(out or "(nvidia-smi produced no output)", flush=True)
                except Exception as _smi:
                    print(f"(nvidia-smi unavailable: {type(_smi).__name__})", flush=True)
                print("-" * 72, flush=True)
                print("  Everything after this point would fail identically, so", flush=True)
                print("  continuing would only waste GPU time. The records already", flush=True)
                print("  written are valid; rerun the same command to resume.", flush=True)
                print("", flush=True)
                print("  A device-side assert is usually one of:", flush=True)
                print("    - a token id outside the vocabulary reaching the embedding", flush=True)
                print("    - multinomial sampling handed NaN/Inf probabilities,", flush=True)
                print("      which quantized weights can produce at high temperature", flush=True)
                print("  Rerun with CUDA_LAUNCH_BLOCKING=1 to get the true line.", flush=True)
                out.write(json.dumps({
                    "uid": r["uid"], "trace": 0, "seed": SEED,
                    "target": r["target"], "truth_key": r["truth_key"],
                    "rxn_class": r.get("rxn_class"),
                    "candidates": [], "raw": "[]", "cfg": CFG,
                    "n_candidates": 0, "k_a": KA, "k_s": KS, "k_b": KB,
                    "n_gen_tokens": 0, "stop_reason": stop,
                    "seconds": round(time.perf_counter() - t1, 3)}) + "\n")
                out.flush()
                os.fsync(out.fileno())
                break

        out.write(json.dumps({
            "uid": r["uid"], "trace": 0, "seed": SEED,
            "target": r["target"], "truth_key": r["truth_key"],
            "rxn_class": r.get("rxn_class"),
            "candidates": cands, "raw": json.dumps(cands),
            "cfg": CFG,
            "n_candidates": len(cands), "k_a": KA, "k_s": KS, "k_b": KB,
            "n_gen_tokens": ntok, "stop_reason": stop,
            "seconds": round(time.perf_counter() - t1, 3)}) + "\n")
        out.flush(); os.fsync(out.fileno())      # never buffer a long run
        tok_total += ntok
        n_done += 1
        # Every 25 molecules OR every two minutes, whichever comes first. A
        # count-only cadence goes silent exactly when a level is pathologically
        # slow -- which is when you most need to see the rate. w8a8 ran for over
        # an hour without printing a single progress line.
        now = time.perf_counter()
        if i % 25 == 0 or i == len(todo) or now - t_last >= 120:
            t_last = now
            el = now - t_run
            _a = torch.cuda.memory_allocated() / 1024**3
            _r = torch.cuda.memory_reserved() / 1024**3
            # PEAK, not just the instantaneous values.
            #
            # This line prints at the END of a molecule, after the beam caches
            # have been released, so live/reserved describe the quiet moment
            # BETWEEN molecules -- not what the card had to hold. With
            # expandable_segments the allocator also returns memory, so
            # reserved moved 10.17 -> 7.77 GB between two consecutive
            # molecules on the same card. Neither number answers "how big a
            # card does this configuration need"; max_memory_allocated does,
            # and it is the only one worth quoting when sizing hardware.
            _pk = torch.cuda.max_memory_allocated() / 1024**3
            _dv = _DRIVER_PEAK[0]
            _tps = (tok_total / el) if el > 0 else 0.0
            print(f"   {i}/{len(todo)}  {el/60:.1f} min  "
                  f"{el/max(1,i):.2f} s/mol  {i/max(el,1e-9):.3f} mol/s  "
                  f"{_tps:.1f} tok/s  mem {_a:.2f}/{_r:.2f} GB live/reserved"
                  f"  peak {_pk:.2f} GB"
                  + (f"  driver peak {_dv:.2f}/{_DEV_TOTAL[0]:.1f} GB"
                     if _dv > 0 else "")
                  + (f"  [{n_err} errors]" if n_err else ""), flush=True)

wall = round(time.perf_counter() - t_run, 1)
meta = {"model": "retrodfm-r-8b", "quant": LEVEL, "task": "retro",
        "engine": "transformers", "parser": "think_answer",
        "quant_detail": detail, "shard": SHARD, "seed": SEED,
        # What this worker actually had to compute, not the whole subset.
        "subset": os.path.basename(SUBSET), "n_shard": len(todo),
        "max_new_tokens": MAX_NEW,
        "k_a": KA, "k_s": KS, "k_b": KB, "alpha": ALPHA,
        "augmented": AUGMENTED,
        "greedy": GREEDY, "cfg": CFG, "base_dtype": BASE_DTYPE,
        "quant_info": QUANT_INFO,
        "temperature": None if GREEDY else TEMP_S,
        "top_p": None if GREEDY else TOP_P,
        "top_k": None if GREEDY else (TOP_K or None),
        "chat_template": bool(getattr(tok, "chat_template", None)),
        "reference_harness": "OpenDFM/RetroDFM-R inference/eval.sh",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_s, "resident_gb": resident,
        "load_peak_gb": load_peak,
        "gen_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        # RESERVED, not allocated. memory_allocated() counts live tensors and is
        # the property of the quantized format; memory_reserved() is what the
        # driver has handed this process and therefore what another job cannot
        # use. They diverge because the allocator keeps freed blocks for reuse,
        # and the staged conversion churns enough differently-sized tensors to
        # strand several GB in partly-used blocks. Reporting only the first
        # overstates how much of the card quantization actually gives back.
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        # THE NUMBER THAT SIZES A CARD. Driver-side peak of (total - free),
        # sampled every QB_MEM_POLL ms. Unlike the allocator figures above it
        # includes the CUDA context and any non-torch allocation, so it is what
        # must be compared against a card's capacity. See _start_mem_sampler().
        "driver_peak_gb": round(_DRIVER_PEAK[0], 2),
        "device_total_gb": round(_DEV_TOTAL[0], 2),
        "mem_poll_ms": _MEM_POLL_MS,
        "ans_chunk": ANS_CHUNK, "gen_chunk": GEN_CHUNK,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        # Without this a fully-resident run and one that streamed half its
        # layers over PCIe are indistinguishable in the results.
        "offload": OFFLOAD, "placement": PLACEMENT,
        "mem_limit_phase": _PHASE,
        "wall_seconds": wall, "n_errors": n_err,
        "mean_seconds_per_molecule": round(wall / max(1, n_done), 3)}
json.dump(meta, open(OUT.replace(".jsonl", ".meta.json"), "w"), indent=2)
print(f"done: {n_done} molecules, {wall/60:.1f} min"
      + (f", {n_err} FAILED (rerun the same command to retry them)"
         if n_err else ""), flush=True)
