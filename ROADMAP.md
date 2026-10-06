# kenAI roadmap — toward GPT-3-level capability

**Objective:** train our own language model from random weights until it reaches the original GPT-3’s level on a documented, broad evaluation suite.

**Status:** planning; the existing character model is the starting prototype. This document does not implement the proposed training system or authorize cloud spending. Initial budget assumption: develop locally until a cloud budget is specified. Research and pricing checked on October 6, 2026.

Green Code’s [I Built an LLM from Scratch](https://www.youtube.com/watch?v=s9w3gtgvNSU) remains the inspiration. The existing local `PLAN.md` describes an optional generation interface; this roadmap governs progress in model capability. Building the interface can support experiments, but it does not replace improvements to data, training, and evaluation.

## 1. Define the destination

Here, “GPT-3” means the original **2020 base language model**, rather than a later instruction-tuned assistant. The published largest model had **175B parameters**, trained on **300B tokens**, with a **2,048-token context window**. Its results included zero-shot and few-shot evaluation. [GPT-3 paper, §2 and §3](https://arxiv.org/html/2005.14165v4)

We target capability rather than requiring the same parameter count. The LLaMA authors reported that their 13B model outperformed GPT-3 on most benchmarks, showing that a smaller, better-trained candidate is a credible research direction. That result does not guarantee our recipe will succeed. [LLaMA paper](https://arxiv.org/abs/2302.13971)

The working destination is therefore a **7B–13B dense model**, with size and token budget selected from our experiments. We may need a different size or more data if evidence warrants it. A literal 175B reproduction is a separate, substantially more expensive alternative, not the default implementation path.

“From scratch” means our candidate’s parameters are randomly initialized and pretrained on our selected corpus. Using PyTorch, a tokenizer library, efficient attention kernels, and distributed training libraries is compatible with that objective. Reference models may be evaluated for comparison; their weights are not used to initialize kenAI.

### What counts as success

1. A versioned evaluation protocol, data manifest, model configuration, and training history accompany the released checkpoint.
2. The base model reaches the historical GPT-3 reference scores on the comparable core tasks below, in both the selected zero-shot and few-shot settings. Record uncertainty and every task result; do not hide regressions behind an average.
3. A broader audit covers reading comprehension, factual recall, in-context adaptation, arithmetic, translation, and sustained text generation. Freeze its task list and thresholds before the candidate’s final run.
4. A separately held-out, contamination-checked audit confirms that improvements extend beyond familiar public benchmarks.
5. Independent reproduction of inference and evaluation works from the saved artifacts.

A passed English benchmark suite supports the statement **“GPT-3-level on this evaluated suite.”** A general equivalence claim requires the broader audit and a clear account of remaining gaps. Personal style and chat behavior are later post-training objectives, evaluated separately from base-model capability.

## 2. Establish evaluation before scaling

These historical reference points come from the GPT-3 paper. They are targets to reproduce under matched protocols, not interchangeable with the defaults of a modern benchmark package. [GPT-3 results, §3 and Appendix H](https://arxiv.org/html/2005.14165v4)

| Task | Published zero-shot | Published few-shot |
| --- | ---: | ---: |
| LAMBADA | 76.2 | 86.4 |
| HellaSwag | 78.9 | 79.3 |
| WinoGrande | 70.2 | 77.7 |
| ARC Easy | 68.8 | 70.1 |
| ARC Challenge | 51.4 | 51.5 |
| OpenBookQA | 57.6 | 65.4 |
| TriviaQA, closed-book | 64.3 | 71.2 |
| CoQA | 81.5 F1 | 85.0 F1 |

Scores are percentages except the explicitly labeled F1 scores. Before using a row as a pass/fail criterion, resolve its dataset revision, split, prompt format, demonstration count and selection, context truncation, answer normalization, and scoring rule against the paper. Some reported results use different splits or test servers. If the exact protocol cannot be reproduced, mark the comparison approximate and do not count it as a verified parity result.

Use a pinned revision of [EleutherAI’s evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) as infrastructure, with explicit task configurations and a kenAI adapter. Do not assume its task defaults reproduce the paper.

Deliverables before a substantial training run:

- `evals/protocol.md`: the frozen comparison contract, including broader-audit criteria.
- `evals/tasks/`: pinned datasets, prompts, metrics, and demonstration seeds.
- `evals/baselines.json`: published references labeled as historical, plus locally reproduced baselines where available.
- `kenai/evaluate.py`: continuation log-likelihood and generation evaluation, without retrieval or external tools for base-model comparisons.
- `reports/`: development results, confidence intervals, error categories, and contamination reports.

Keep development evaluations separate from the final audit. Use small development subsets during iteration, then the full frozen suite for release. Report 95% bootstrap intervals and variation across fixed demonstration seeds where applicable. If results are too uncertain to support parity, report that outcome rather than declaring success. Access to the original GPT-3 model is not assumed; a historical comparison is not a new side-by-side experiment.

The current Shakespeare loss cannot establish GPT-3-level ability. Its character-level loss also cannot be compared directly with subword-token loss. Compare models on the same tokenizer/corpus, or use a consistently defined text-level measure such as bits per byte when tokenizers differ.

## 3. Starting point and hardware roles

Today’s implementation has a working 826,368-parameter character GPT, local training, checkpointing, and generation. It does not yet have the proposed subword tokenizer, streaming corpus, token-based training schedule, distributed trainer, or capability evaluation suite.

The M2 Mac with 32 GB of unified memory is our **development machine**: implement and test the pipeline, inspect samples, prepare limited data shards, and run small models. Begin with 20M–50M parameter development configurations and short token budgets; measure memory and speed before increasing them. Some larger configurations may fit, but fitting in memory does not make full pretraining fast enough.

For a rough memory budget, assume about **16 bytes per parameter** for conventional weights, gradients, and Adam training state, before activations and runtime overhead. This gives about 16 GB for 1B parameters, 112 GB for 7B, and 208 GB for 13B. The actual precision and optimizer implementation change this estimate. Inference memory is a different calculation.

Use cloud CUDA GPUs for substantial pretraining. Start with one suitable GPU for pilots, then a connected multi-GPU node, and only move to multiple nodes after measuring scaling. Never derive a completion date for a large run from the existing Shakespeare model’s speed.

## 4. Development stages and decision gates

The sizes and token budgets below are experiment proposals. They are not promises that a particular model size will achieve a capability level. B = billion; T = trillion. Future token counts refer to our frozen subword tokenizer, not characters or a dataset provider’s tokenizer.

| Stage | Model and data proposal | Main work | Evidence required to advance |
| --- | --- | --- | --- |
| 0 — Measurement | Current model and tiny fixtures | Freeze the evaluation contract; establish reproducible baselines and an experiment log. | Evaluator passes hand-checked scoring cases; task provenance and metrics are recorded. |
| 1 — New training foundation | 20M–50M model; 50M–200M tokens for development | Add byte-level BPE, document-based data preparation, sharded batches, and a versioned model/checkpoint format. | Unicode round trips, no split leakage, falling held-out loss, correct restart behavior, and measured Mac memory/throughput. |
| 2 — Small general-text model | Approximately 110M–350M; 2B–7B tokens | Compare data mixtures and schedules; establish a stable single-GPU CUDA trainer. | A fixed-compute experiment improves validation and development tasks; the full candidate run fits the agreed budget. |
| 3 — Scaling pilot | 1B–3B; 20B–60B tokens | Add sharded training, measure multi-GPU efficiency, and test restart/failure handling. | Measured scaling trends, stable runs, and benchmark progress justify a larger candidate. These pilots are not expected to equal GPT-3. |
| 4 — Capability candidate | 7B–13B; initial 140B–260B tokens, with a separately planned extension toward 0.5T–1T or more | Select the architecture and data recipe from prior evidence; run pretraining and evaluation at scheduled token milestones. | Each extension is supported by learning curves, task results, and a refreshed compute estimate. Stop scaling blindly if gains flatten. |
| 5 — Final evaluation | Best qualified base checkpoint | Run the frozen benchmark suite and broader audit; reproduce results and publish limitations. | Meet the success criteria in sections 1–2. Otherwise identify the gaps and propose the next targeted experiment. |
| 6 — Optional personal assistant | A copy of the qualified base model | Instruction tuning and personal-style training on suitable examples. | Separate held-out instruction/style evaluations improve without unacceptable base-capability regressions. |

Treat roughly 20 training tokens per parameter as an initial scaling experiment, not a sufficiency rule. The Chinchilla research motivates balancing model size and training tokens under a compute budget; downstream quality and serving requirements may justify much longer training. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

Increasing model size normally starts a new training run. Continuing a checkpoint with the same architecture and more tokens is a separate operation. Do not assume the current character checkpoint can be enlarged into the future model.

## 5. The first engineering milestone

Build a reliable subword training system before buying substantial compute. Keep the existing character implementation usable as an educational reference.

### Tokenizer and corpus

- Train a **byte-level BPE tokenizer** on a representative sample of training documents only. Test 16K and 32K vocabulary sizes; freeze the selected tokenizer before comparative runs.
- Support arbitrary UTF-8 text and explicit document-end tokens. Test exact round trips and checkpoint compatibility. A library implementation can be used and verified against small, understandable examples. [Tokenizer training documentation](https://huggingface.co/docs/tokenizers/quicktour)
- Save tokenizer files and a content hash alongside every checkpoint. Changing the vocabulary invalidates a direct resume.
- Replace the current whole-corpus in-memory batching with token shards and deterministic sampling. Separate document groups before packing sequences, then pack with explicit boundaries and documented attention behavior.

### Model and training loop

- Keep a dense decoder-only Transformer. Start from a simple configuration; evaluate changes such as rotary positions, RMSNorm, and a gated feed-forward layer individually under fixed compute rather than changing everything at once.
- Use a proposed 12-layer, width-768, 12-head, 32K-vocabulary configuration with tied embeddings for the approximately 110M stage. Calculate the actual parameter count from the implementation. Begin at 1,024 tokens and validate 2,048-token training before parity comparisons.
- Add efficient scaled-dot-product attention with a tested reference implementation. Feature-test device support; preserve a working CPU/MPS path.
- Make `max_tokens`, tokens per optimizer step, gradient accumulation, warmup, learning-rate decay, weight decay, and gradient clipping explicit configuration values. Log effective tokens processed instead of treating step counts as comparable across runs.
- Establish an FP32 correctness baseline. Add supported mixed precision on CUDA after parity checks; validate any Mac precision mode separately.
- Save model/optimizer/scheduler/scaler state, tokenizer/data hashes, data cursor, device RNG state, configuration, code commit, and consumed-token count. Test interrupted versus uninterrupted runs on the same backend; describe limits across backends.
- Track throughput, peak memory, validation loss by source, gradient norms, checkpoint duration, and estimated completion cost. Stop or recover clearly on non-finite loss or corrupt data.

### Compatibility

Introduce a versioned tokenizer/model interface and a new checkpoint schema. Preserve loading of the existing character model. Add an adapter for the generation interface planned in local `PLAN.md`; its character-specific controls will need to distinguish subword tokens from characters. A UI is useful for inspection, while the evaluator remains the basis for capability claims.

## 6. Data plan

There is enough public material to begin without personal writing. Start with a bounded, streamed sample of **FineWeb-Edu**, then compare broader text mixtures in small experiments. Its `sample-10BT` configuration is described as approximately 10B GPT-2 tokens; recount tokens with our tokenizer. Pin a dataset revision rather than silently following updates. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

Before scaling the corpus:

1. Record source, revision, document identifier, collection date where available, license/terms metadata, and preprocessing configuration. The dataset card lists ODC-By and Common Crawl terms; do not label every source document public domain.
2. Normalize and filter broken text, boilerplate, duplication, and low-quality material. Inspect samples from each source and quality bucket; record acceptance rates.
3. Group exact and near-duplicate documents before splitting. Allocate stable training, development, and final-audit document sets; do not reuse the prototype’s single-file tail split for web-scale data.
4. Remove detected benchmark overlap from training and log the matching method and removed counts. Record residual contamination uncertainty; a filter is not proof of zero overlap.
5. Tokenize into checksummed shards with deterministic ordering, source labels, and a resumable manifest. Count unique tokens/documents separately from total token presentations across repeated passes.
6. Compare data mixtures at the same model size and token budget. Add general prose, reference material, mathematical text, code, or multilingual content when the audit identifies a relevant gap and usable sources are available.

At 32K vocabulary, two-byte token IDs are possible if all reserved IDs fit the format. **1T tokens then require roughly 2 TB for token IDs alone**; text, indexes, manifests, temporary preprocessing, replicas, and checkpoints add more. Stream or process bounded shards on the Mac. Plan cloud/object storage and data-loader throughput before the large run.

Personal writing belongs in a later style dataset. It cannot replace the broad corpus needed for general language capability.

## 7. Distributed training and artifact storage

After the single-GPU trainer is correct, use a maintained sharding implementation such as [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). Test optimizer-state sharding, distributed checkpointing, gradient accumulation, and one-GPU versus multi-GPU behavior on a small model first. Add activation checkpointing when measured memory requires it.

For each hardware proposal, run a short benchmark with the actual model, context, precision, batch, and dataset loader. Report total cluster tokens/second, memory, interconnect, restart time, and dollars per billion tokens. Large multi-node allocations require a cluster quote and verified interconnect; an isolated GPU’s hourly price does not establish cluster availability or performance.

Keep source, configs, manifests, evaluation results, and model cards in Git. Keep future large datasets and checkpoints in versioned artifact storage or an appropriate model repository, referenced by hash. Preserve the existing small bundled demo; do not start committing multi-gigabyte training artifacts to ordinary Git history. Verify a restore to a fresh machine before extending a long run.

## 8. Compute and budget model

For initial dense-transformer planning, use:

```text
training FLOPs ≈ 6 × parameter count × training tokens
GPU-hours ≈ training FLOPs / (sustained FLOPs per GPU-second × 3,600)
compute cost ≈ GPU-hours × price per GPU-hour
```

This approximation omits effects such as attention overhead and implementation-specific recomputation. It must be replaced by measured end-to-end throughput before spending decisions. It does not predict capability.

The scenarios below assume **200 TFLOP/s sustained per GPU** and **$4 per GPU-hour**. These are planning inputs, not kenAI measurements. For price context, Lambda’s public page listed H100 SXM at $3.99 per GPU-hour in an eight-GPU instance and $4.29 for a single-GPU instance when checked. Multi-node quotes, availability, and taxes can differ. [Lambda instance pricing](https://lambda.ai/instances)

| Scenario | Training tokens | Approx. training FLOPs | GPU-hours at the assumed rate | Compute-only cost |
| --- | ---: | ---: | ---: | ---: |
| 1B scaling pilot | 20B | 1.20 × 10²⁰ | 167 | $667 |
| 3B scaling pilot | 60B | 1.08 × 10²¹ | 1,500 | $6,000 |
| 7B candidate | 1T | 4.20 × 10²² | 58,333 | $233,333 |
| 13B candidate | 1T | 7.80 × 10²² | 108,333 | $433,333 |
| Literal 175B reference | 300B | 3.15 × 10²³ | 437,500 | $1,750,000 |

These are single-run arithmetic scenarios, not total project budgets or vendor quotes. Small models can achieve substantially less utilization than assumed. At 100–400 sustained TFLOP/s and the same $4 rate, the 7B scenario spans approximately **$117K–$467K**, and the 13B scenario approximately **$217K–$867K**. Extra token presentations increase those costs proportionally.

Budget separately for preprocessing, storage, transfers where charged, evaluation, idle allocation time, failed runs, ablations, and engineering time. For an initial research funding discussion, reserve two to three times a selected run’s compute estimate for the experimental campaign, then replace that allowance with actual measured line items.

At the same sustained-rate assumption, 64 GPUs would take about 38 days for the 7B/1T scenario or 71 days for 13B/1T. Eight GPUs would take roughly 304 or 564 days. These are conditional arithmetic estimates; interconnect, hardware availability, and achieved efficiency determine real elapsed time.

### How to proceed at different initial budgets

| Initial cloud budget | Reasonable planning scope |
| --- | --- |
| $0 | Implement and test stages 0–1 locally; measure the Mac and prepare bounded data. No GPT-3-level training deadline. |
| Up to $1,000 | Add metered CUDA pilots and a limited stage-2 campaign after throughput measurements. Do not promise a 1B completed run from the raw table alone. |
| Up to $10,000 | Consider repeated small experiments and a stage-3 pilot where quotes fit; retain reserve for failures and evaluation. |
| Substantial research funding | Plan a 7B–13B campaign with a vetted dataset, cluster quote, measured scaling, and evaluation protocol. Assess the result against the capability gate. |

The 32 GB Mac supports the project’s development work. It does not remove the compute cost of the final target. If future funding is limited, finish and release the strongest measured smaller model while keeping the GPT-3-level objective explicitly unmet.

## 9. Immediate work plan

For one builder, allow roughly four to six focused engineering weeks for the first foundation milestone, with extra time for learning and data issues. This is a sequencing estimate, not a promise about training speed or the final research outcome.

| Order | Deliverable | Completion check |
| --- | --- | --- |
| Week 1 | Evaluation specification, historical baselines, experiment manifests, data-source sample | Scoring fixtures are correct; comparison limitations are explicit; data revision and development splits are fixed. |
| Week 2 | BPE tokenizer and document/shard pipeline | Unicode round trips, deterministic batches, split separation, and restart cursor tests pass. |
| Week 3 | Versioned model/trainer and 20M–50M configuration | Tiny-corpus overfit, finite gradients, causal masking, checkpoint restore, and existing-model compatibility pass. |
| Week 4 | Local development run and performance report | Reproducible loss curves, sample review, throughput/memory measurements, and an estimated cost per billion tokens exist. |
| Weeks 5–6 if needed | CUDA pilot preparation, evaluation adapter, data-mixture comparison | A short measured pilot establishes whether the next full experiment is affordable and technically sound. |

Do not schedule the final capability milestone until the pilot provides actual throughput, cost, and learning-curve evidence. It is a multi-stage research effort, not a guaranteed six-week build.

### Proposed repository work

All paths below are planned unless already present:

- `configs/`: versioned 30M, approximately 110M, and later scaling configurations.
- `kenai/tokenizers/`: character compatibility and a byte-level BPE implementation/adapter.
- `kenai/data/`: document manifests, filtering, tokenization, sharding, and batching.
- `kenai/training/`: token-based schedules, precision, logging, and checkpoint state.
- `kenai/evaluate.py`, `evals/`, and `reports/`: reproducible capability measurement.
- `tests/`: causal correctness, data leakage checks, loss/gradient equivalence, and restart tests appropriate to each backend.

**First implementation task:** establish the evaluation contract and dataset manifest, then build the BPE/data pipeline with tests. Keep this milestone bounded; no large download or paid training run is needed to start it.

## 10. Progress record

- [x] Character-model prototype trained and published.
- [x] Capability roadmap and research references documented.
- [ ] Initial cloud budget selected.
- [ ] Evaluation protocol and broader audit frozen.
- [ ] BPE tokenizer and corpus pipeline implemented.
- [ ] Local subword-model milestone completed.
- [ ] Single-GPU CUDA campaign measured.
- [ ] Distributed pilot validated.
- [ ] Final candidate funded and trained.
- [ ] GPT-3-level capability demonstrated under the stated evaluation protocol.

Update this record with links to measured results as work lands. A larger parameter count, lower training loss, or polished interface alone does not complete the final objective.
