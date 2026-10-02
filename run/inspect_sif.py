#!/usr/bin/env python3
"""What is actually inside a .sif -- without singularity, without a GPU.

    python3 inspect_sif.py                 searches for every .sif and reads it
    python3 inspect_sif.py some.sif        one specific image
    python3 inspect_sif.py ~/images        every .sif in a directory

WHY THIS EXISTS
---------------
Two .sif files look identical from the outside, and running the wrong one
fails a different way in every level with the real cause buried in a warning.
`qb.sh` prints a `build :` line for exactly this reason -- but that needs
singularity and a copy of the image you can execute, and it says nothing at all
for an older image with no /opt/qb/build-versions.json.

A .sif is a SquashFS filesystem behind a small header, so the package inventory
can be read straight off the disk: no singularity, no GPU, no root, and it
works on a login node or a laptop that cannot run the image at all.

An image with a name that reads like the RetroDFM one but without torchao,
bitsandbytes or transformers cannot run any RetroDFM level. That is the
mistake this script makes visible in one command.
"""
import os
import re
import struct
import sys

KEY = ("torch", "torchao", "bitsandbytes", "transformers", "accelerate",
       "rdkit", "dgl", "syntheseus", "vllm", "sglang", "llama_cpp_python",
       "peft", "safetensors")
DIST = re.compile(r"^([A-Za-z0-9_.\-]+?)-([0-9][^-]*?)\.(?:dist|egg)-info$")


def squashfs_offset(path):
    """The primary SquashFS partition inside the SIF container.

    Parsed from the descriptor block rather than guessed, then checked against
    a magic scan of the first few MB, because a wrong offset produces a
    confident-looking empty listing rather than an error.
    """
    with open(path, "rb") as f:
        head = f.read(4 * 1024 * 1024)
        if head[32:41] != b"SIF_MAGIC":
            # not a SIF -- maybe a bare squashfs
            return 0 if head[:4] == b"hsqs" else None
        try:
            f.seek(32 + 10 + 3 + 3 + 16)
            (_ct, _mt, _df, _dt, descroff, descrlen,
             dataoff, datalen) = struct.unpack("<qqqqqqqq", f.read(64))
            f.seek(descroff)
            raw = f.read(descrlen)
            for i in range(0, max(0, len(raw) - 8), 4):
                off, = struct.unpack_from("<q", raw, i)
                if 1000 < off < dataoff + datalen:
                    f.seek(off)
                    if f.read(4) == b"hsqs":
                        return off
        except Exception:
            pass
    scan = head.find(b"hsqs")
    return scan if scan > 0 else None


def inspect(path):
    print("=" * 78)
    print(os.path.abspath(path))
    sz = os.path.getsize(path)
    import datetime
    print("   %.1f GB   %s" % (sz / 1e9, datetime.datetime.fromtimestamp(
        os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")))
    off = squashfs_offset(path)
    if off is None:
        print("   not a SIF or squashfs image -- nothing to read")
        return
    try:
        from PySquashfsImage import SquashFsImage
    except ImportError:
        sys.exit("needs PySquashfsImage:\n    pip install PySquashfsImage")
    try:
        img = SquashFsImage.from_file(path, offset=off)
    except TypeError:
        img = SquashFsImage(path, offset=off)

    envs, qb, harness = {}, set(), set()
    try:
        for f in img:
            p = f.path
            if "/site-packages/" in p:
                env = (p.split("/envs/")[1].split("/")[0]
                       if "/envs/" in p else "system")
                tail = p.split("/site-packages/", 1)[1].split("/")[0]
                m = DIST.match(tail)
                if m:
                    envs.setdefault(env, {})[m.group(1).lower().replace("-", "_")] = m.group(2)
                else:
                    envs.setdefault(env, {}).setdefault(
                        tail.lower().replace("-", "_"), None)
            elif p.startswith("/opt/qb/"):
                rel = p[len("/opt/qb/"):]
                if "/" not in rel.rstrip("/"):
                    qb.add(rel)
                base = os.path.basename(p)
                if base in ("run_llm_cluster.sh", "run_llm.py", "qb.sh",
                            "score_sweep.py", "probe_capability.py",
                            "build-versions.json"):
                    harness.add(base)
    finally:
        img.close()

    print()
    if not envs:
        print("   no python environment found")
    for env in sorted(envs):
        pk = envs[env]
        found = {k: v for k, v in pk.items() if k in KEY}
        print("   env %-10s %4d packages" % (env, len(pk)))
        for k in KEY:
            if k in found:
                print("        %-18s %s" % (k, found[k] or "(version unknown)"))

    print()
    print("   /opt/qb        %s" % (", ".join(sorted(qb)) if qb else "ABSENT"))
    print("   harness files  %s" % (", ".join(sorted(harness)) if harness
                                    else "none baked in"))

    # ---- verdict ---------------------------------------------------------
    print()
    best = max(envs.items(), key=lambda kv: len(kv[1]))[1] if envs else {}
    ao = best.get("torchao")
    bnb = "bitsandbytes" in best
    tf = "transformers" in best
    tv = best.get("torch") or ""
    print("   CAN THIS IMAGE RUN THE RetroDFM LEVELS?")
    if not tf:
        print("      NO -- transformers is not installed. This is not the")
        print("      RetroDFM image whatever the file is called.")
    else:
        print("      transformers present, torch %s" % (tv or "?"))
    print("      bitsandbytes (nf4 / nf4dq / fp4) : %s"
          % ("yes, %s" % best.get("bitsandbytes") if bnb else "NO"))
    if not ao:
        print("      torchao levels                   : NO -- torchao absent")
        print("      2-bit                            : NO")
    else:
        print("      torchao levels                   : yes, %s" % ao)
        try:
            major, minor = (int(x) for x in ao.split(".")[:2])
        except Exception:
            major = minor = -1
        tmaj, tmin = 0, 0
        m = re.match(r"(\d+)\.(\d+)", tv)
        if m:
            tmaj, tmin = int(m.group(1)), int(m.group(2))
        if (major, minor) >= (0, 14):
            ok = (tmaj, tmin) >= (2, 6)
            print("      2-bit                            : IntxWeightOnlyConfig"
                  " (weight_dtype=torch.int2)")
            print("        torch >= 2.6 for the int2 dtype : %s"
                  % ("yes" if ok else "NO -- int2_ao will fall back to uint2, "
                     "and w2a8_intx will not resolve"))
        else:
            print("      2-bit                            : UIntXWeightOnlyConfig"
                  " (dtype=torch.uint2)")
            print("        w2a8_intx needs torchao >= 0.14 : NO, int2_ao only")


SKIP = {".git", "node_modules", "__pycache__", ".cache", ".conda", "anaconda3",
        "miniconda3", "site-packages", ".vscode-server", "snap", "proc", "sys",
        "dev", "AppData", "Windows", "Program Files", "Program Files (x86)",
        "$Recycle.Bin", ".local", ".npm", "Zotero", "OneDrive"}


def find_sifs(roots, max_depth=7):
    """Every .sif under these roots.

    A .sif is large and there are never many, so a bounded walk finds them
    faster than anyone can type a path -- and typing the path is what has gone
    wrong here repeatedly. Heavy directories are skipped by name so this stays
    a few seconds even from a home directory.
    """
    out = []
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                    onerror=lambda e: None):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP and not d.startswith(".cache")]
            for fn in filenames:
                if fn.endswith(".sif"):
                    out.append(os.path.join(dirpath, fn))
    return out


def main():
    args = sys.argv[1:]
    paths = []
    if not args:
        print("searching for .sif images ...", flush=True)
        paths = find_sifs([".", "~", "/opt", "/srv", "/scratch",
                           "/mnt/c/Users", "/media"])
        if not paths:
            sys.exit("no .sif found under the current directory, your home "
                     "directory, /opt, /srv, /scratch, /mnt/c/Users or /media.\n"
                     "Pass one:  python3 inspect_sif.py /path/to/image.sif")
        print("found %d:" % len(paths))
        for p in paths:
            print("   %s" % p)
        print()
    for a in args:
        if os.path.isdir(a):
            import glob
            paths += sorted(glob.glob(os.path.join(a, "*.sif")))
        else:
            paths.append(a)
    seen = set()
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen or not os.path.exists(p):
            continue
        seen.add(rp)
        inspect(p)
        print()


if __name__ == "__main__":
    main()
