# Generating the training questions on a rented GPU

One paid step stands between the raw filings and a fine-tuned reranker: an open
model has to read each sampled table group and write a Vietnamese question those
tables answer. Everything before it is deterministic and runs locally;
everything after it is a filter.

The groups it reads are written by `scripts/sample_groups.py`, and every
proportion that script conditions on is printed by
`scripts/measure_questions.py` from the public questions and the public corpus.
Run that first if you want to check a constant rather than trust it.

## The box

A plain PyTorch/CUDA template, any 24 GB card. A 4090 at about $0.35/hour is the
cheapest thing that fits. Ask for **60 GB of disk**: Qwen2.5-14B-Instruct is a
28 GB download in fp16 and is quantized to NF4 on load, so the download, not the
resident model, sets the disk requirement. Prefer an instance with a high
measured download speed — the meter runs while the weights arrive.

vLLM buys nothing here. The generation is one pass of batched sampling through
`transformers`, and a new serving path would cost more to wire up than it saves.

## Run it

Take the host and port from the instance's Connect button, then:

```
bash vast/launch.sh root@ssh4.vast.ai 41521
```

That ships `tasks.jsonl` and `generate_questions.py`, installs the three packages
the template lacks, starts the run under `nohup`, and tails the log. Detaching
with ctrl-c does not stop it. Reconnecting and re-running the same command
resumes from the questions already written.

The default is 6,000 of the 12,488 sampled groups, which is roughly $1 and about
three hours. `GROUPS=all bash vast/launch.sh ...` sends the whole set. The subset
is drawn with a seeded shuffle rather than `head`, because `tasks.jsonl` is
ordered by report and the first lines are one alphabetical slice of the corpus.

A group carries one to ten tables, so the prompts are not one length. The
generator shows each table up to 1,100 characters but caps the group at 6,000,
which is what keeps a ten-issuer comparison inside the 3,072-token input window;
batches are formed by table count so padding does not blow up on the mix.

Watch the rate the script prints after the first batch. Below about 1.5
questions per second, the box is slower than it should be for a 4090 and the run
will not pay for itself.

When it finishes:

```
bash vast/fetch.sh root@ssh4.vast.ai 41521
```

which copies the questions back and runs the audit — the surface-feature
comparison against the released questions, reported as a total variation
distance per feature. Read that before assembling anything. Then, locally:

```
.venv/bin/python scripts/assemble_training.py
```

## What each piece is for

| file | role |
|---|---|
| `launch.sh` | local; subsets, ships, starts |
| `generate.sh` | on the box; installs, runs under nohup, tails |
| `fetch.sh` | local; copies the questions home and audits them |

The repository is never cloned onto the box. `generate_questions.py` imports
only `torch` and `transformers`, so two files are the whole payload.

## Settings that matter

They live at the top of `kaggle/generate_questions.py`, and they are what the
methodology section has to state:

```
MODEL_NAME   Qwen/Qwen2.5-14B-Instruct    the 14B open-weight ceiling the organizers allow
QUANTIZATION nf4                          sampling noise at T=0.7 dominates quantization noise
TEMPERATURE  0.7                          Promptagator's setting
SEED         20260818
```

On a free Kaggle T4, drop to `Qwen/Qwen2.5-7B-Instruct` first; 14B in NF4 fits
16 GB but leaves no room for a batch of sixteen.
