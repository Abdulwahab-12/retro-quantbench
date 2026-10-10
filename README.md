# Post-training quantization effect on a reasoning language model for retrosynthesis

Code, predictions and analysis scripts for the paper by Abdulwahab Hussein and Igor V. Tetko.

RetroDFM-R-8B ([Zhang et al.](https://doi.org/10.48550/arXiv.2507.17448)) was evaluated on the 5005 reactions of the USPTO-50K test split at its original precision and with four post-training quantizations applied while the model is loaded.

| Level | Weights | Library |
|---|---|---|
| bf16 | 16-bit brain floating point (original model) | – |
| int8_ao | 8-bit integer, weight only | torchao `Int8WeightOnlyConfig` |
| nf4 | 4-bit NormalFloat | bitsandbytes |
| fp4 | 4-bit floating point | bitsandbytes |
| nf4dq | 4-bit NormalFloat, scales quantized to 8 bits | bitsandbytes |

Two inference settings of the RetroDFM-R authors were used:

| Setting | k_a (product SMILES) | k_s (reasoning paths per SMILES) | k_b (answers per path) |
|---|---|---|---|
| Default | 1 | 1 | 1 |
| Augmented | 20 | 10 | 10 |

In the augmented setting the k_a SMILES are the canonical SMILES and SMILES rooted at other atoms (fewer than 20 when the molecule has fewer distinct starting atoms). Reasoning paths are sampled at temperature 1.1 and answers at temperature 1.4, and the candidates are ranked by how often they are predicted. The default setting samples one answer at temperature 1.0. Both settings use top-p 1.0, no top-k and at most 2048 new tokens.

## Results

Table 2 of the paper. Top-k accuracy (%) on the USPTO-50K test split, n = 5005.

| Level | Default Top-1 | Augmented Top-1 | Top-3 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| bf16 | 59.8 | 64.8 | 84.2 | 88.8 | 91.5 |
| int8_ao | 60.0 | 64.5 | 84.5 | 88.9 | 91.4 |
| fp4 | 55.6 | 61.8 | 83.0 | 88.4 | 91.7 |
| nf4 | 57.7 | 63.0 | 84.2 | 88.9 | 92.3 |
| nf4dq | 57.9 | 63.2 | 84.0 | 88.9 | 92.1 |

The 95% confidence interval of every Top-1 value is about ±1.4; all intervals are in `results/tables/accuracy.csv`.

Because all levels answer the same reactions, each quantized level is compared with bf16 reaction by reaction: McNemar's exact test with Holm correction within each setting (4 Top-1 tests in the default setting, 16 tests in the augmented setting), the paired difference with its 95% interval, and a two one-sided test (TOST, 90% interval) for equivalence within ±1 percentage point (`results/tables/paired_differences.csv`, Table S2 of the paper).

Further results in `results/tables/`:

| File | Content |
|---|---|
| `seeds.csv` | bf16 in the default setting with seeds 1234, 695 and 1066: Top-1 and the number of reactions whose result changed |
| `distinct_candidates.csv` | distinct candidates per product in the augmented setting; at least 10 for 35.3% of the products with bf16, 35.0% with int8_ao, 66.6% with fp4, 51.7% with nf4 and 51.4% with nf4dq |
| `top10_restricted.csv` | Top-k on the 1636 reactions for which bf16, fp4, nf4 and nf4dq all returned at least 10 distinct candidates |
| `accuracy.csv` | every prediction file, including bf16 and nf4 with k_a = 20, k_s = 1, k_b = 10 |

## Repository layout

```
make_tables.py              rebuilds the tables in results/tables from the predictions
requirements.txt            Python packages needed by make_tables.py
run/                        everything needed to run the model; the image goes here too
  qb.sh                     starts a run inside the Singularity image
  run_llm_cluster.sh        splits the products over the GPUs and merges the results
  qb-run_llm.py             loads and quantizes the model and runs the inference
  score_sweep.py            top-k accuracy
  paired_stats.py           paired comparisons with bf16 (McNemar, Holm, TOST)
  distinct_candidates.py    distinct candidates per product
  probe_capability.py       checks which quantization kernels run on the card
  chat_template.jinja       the RetroDFM-R chat template
  check_sif.sh              checks that the image can drive the card
  qb_power.sh               GPU energy per product
  qb_memcheck.sh            GPU memory per level
  throughput.sh             seconds and tokens per product from finished runs
docker/
  Dockerfile.llm            the image
  build_images.sh           builds the image and converts it to Singularity
data/
  subset_full.jsonl         the 5005 test reactions
results/
  predictions/              one file per run (level, setting and seed in the name)
  tables/                   CSV files written by make_tables.py
```

## Rebuilding the tables without a GPU

```
pip install -r requirements.txt
python make_tables.py
```

This takes about five minutes. The script decompresses the predictions, scores them with `run/score_sweep.py`, compares them with bf16 with `run/paired_stats.py`, counts candidates with `run/distinct_candidates.py` and writes the CSV files to `results/tables/`.

A prediction counts as correct when its reactant set matches the ground truth in any fragment order, after charges in both have been neutralized with RDKit's Uncharger and the SMILES canonicalized with RDKit. Invalid SMILES are removed and duplicates merged before the ranks are counted.

## Running the model

All files in `run/` work together from one folder, and the image has to be in that folder too. `qb.sh` finds the image next to itself and uses the scripts beside it in place of the copies inside the image. To run on another machine, copy the whole `run/` folder with the image inside it.

This needs a Linux machine or WSL with an NVIDIA GPU (Turing or newer) and Singularity or Apptainer. Docker is needed only to build the image. The RetroDFM-R-8B weights are not included; they are available from the [RetroDFM-R repository](https://github.com/OpenDFM/RetroDFM-R).

Build the image and put it in `run/`:

```
bash docker/build_images.sh --mslk --model-dir /path/to/RetroDFM-R-8B
cp ~/retro-llm-mslk.sif run/
```

The image contains the model, the test set and the scripts, so the machine that runs it needs no Python environment and no network. The same image runs both settings.

Check that the image can drive the card, then run the five levels in each setting:

```
cd run
bash check_sif.sh --deep

# default setting
bash qb.sh --levels bf16,int8_ao,nf4,fp4,nf4dq --gpus 4

# augmented setting
bash qb.sh --levels bf16,int8_ao,nf4,fp4,nf4dq --ka 20 --ks 10 --kb 10 --gpus 4
```

On a card with less than 20 GB, such as an 11 GB RTX 2080 Ti, the same commands work with `--gpus 1`, for example `bash qb.sh --levels int8_ao --ka 1 --ks 1 --kb 1 --gpus 1`. int8_ao is then converted to int8 in system RAM and only the finished int8 model is moved to the card, so the card never holds the 16-bit weights; this needs about 16 GB of free system RAM and two minutes of loading. `QB_CONVERT_ON_CPU=1` does the same on a larger card. The 4-bit levels are quantized while loading on any card.

`--compiled 1` compiles the decoding step with `torch.compile`, for example `bash qb.sh --levels int8_ao --ka 1 --ks 1 --kb 1 --compiled 1`; `--compiled 0` (the default) runs without it. Weights, prompts and decoding settings are the same; compiled runs are written to their own files (`int8_ao.compiled`), so they never mix with or continue an uncompiled run. The compilation warm-up happens before the first product and is not counted in the seconds per product. Only int8_ao is compiled: bf16 was slower compiled than uncompiled, and the 4-bit levels cannot be compiled (transformers does not support it for bitsandbytes), so they always run uncompiled. Compiling helps int8_ao in the default setting (one sequence at a time); in the augmented setting, where ten sequences are decoded together, compiled int8_ao is slower than uncompiled. On machines with several GPUs of less than 20 GB each, the workers convert int8_ao in system RAM one after another, so only one 16 GB copy is held at a time.

Predictions are written to `run/results/raw/` and scored at the end of the run. The scores are printed after charge standardisation (see above). `bash qb.sh --score-only` scores them again, and repeating a command continues an interrupted run where it stopped. The augmented setting takes 6 to 9 minutes per product and level on one RTX 5090 or RTX PRO 5000.

Each product is sampled with the seed 1234 plus its position in the list of the GPU that runs it, so the individual predictions depend on how the products are split over the GPUs and on restarts. A new run is statistically equivalent to the stored one, not identical. Every record keeps its generation settings in the `cfg` field.

Speed, memory and energy depend on the card and are measured on your own hardware, from inside `run/`:

```
bash qb_power.sh --out /path/to/empty/folder   # GPU energy per product
bash qb_memcheck.sh                            # live and peak GPU memory per level
bash throughput.sh results                     # seconds and tokens per product
```

The options of each script are listed at the top of the file.

## Data

`data/subset_full.jsonl` holds the USPTO-50K test split of [Coley et al. (2017)](https://doi.org/10.1021/acscentsci.7b00355), the split distributed with [GLN](https://github.com/Hanjun-Dai/GLN). The split has 5007 rows, two of which repeat another reaction, so the file contains 5005 reactions. Each line holds `uid`, `target` (product SMILES), `truth_key` (reactant SMILES), `rxn_class` (reaction class) and `source_id` (patent number).

`results/predictions/` holds the five levels in the default and augmented settings, bf16 in the default setting with the seeds 695 and 1066 (`.seed0695`, `.seed1066`; the other files use 1234), and bf16 and nf4 with k_a = 20, k_s = 1, k_b = 10 (`.k20x1x10`). Every prediction file covers all 5005 reactions. Each line of a prediction file is one product:

| Field | Content |
|---|---|
| `uid`, `target`, `truth_key`, `rxn_class` | as in `data/subset_full.jsonl` |
| `candidates` | predicted reactant sets, best first (`raw` holds the same list as a string) |
| `n_candidates` | length of that list |
| `k_a`, `k_s`, `k_b` | inference setting |
| `cfg` | generation settings |
| `seed` | base seed |
| `trace` | always 0 |
| `n_gen_tokens` | tokens generated for the product |
| `seconds` | time spent on the product |
| `stop_reason` | `eos` for every stored record |

## Hardware and software

Predictions were collected on NVIDIA GeForce RTX 5090 (32 GB), RTX PRO 5000 Blackwell (48 GB) and GeForce RTX 3090 (24 GB) cards; speed and energy were measured on an RTX 5090 and an RTX 2080 Ti (11 GB). The image built by `docker/build_images.sh --mslk` contains Python 3.12, torch 2.13.0+cu132, transformers 4.57.6, torchao 0.18.0+cu132, bitsandbytes 0.50.2 and MSLK 1.3.0.

## License

The code is released under the MIT license (`LICENSE`). The data and results in `data/` and `results/` are released under CC BY 4.0 (`LICENSE-DATA`). The RetroDFM-R model and the USPTO-50K reactions remain under their own terms, and the model weights are not part of this repository.

## Citation

Hussein, A.; Tetko, I. V. Post-training quantization effect on a reasoning language model for retrosynthesis. See `CITATION.cff`.
