# kenAI — building an LLM from scratch

An open project to learn how language models work by building and training one from the ground up. Start with text, turn it into tokens, build a Transformer, and teach it to predict what comes next.

Inspired by Green Code’s [**I Built an LLM from Scratch**](https://www.youtube.com/watch?v=s9w3gtgvNSU).

The first milestone is a small GPT-style model that can train locally. Its weights start randomly, its character vocabulary comes from the dataset, and its predictions improve through training. PyTorch provides tensor operations and automatic differentiation; the tokenizer, attention mechanism, Transformer blocks, training loop, and generation pipeline are implemented in this repository.

## What we’re building

kenAI is a **decoder-only Transformer** that learns to continue text, one character at a time. It is a small starting point for understanding LLMs: the current implementation is an educational text generator, and useful conversation or personal writing style would require further data and training.

The project includes:

- A character tokenizer built from the input text.
- Learned character and position embeddings.
- Multi-head causal self-attention, feed-forward layers, and residual connections.
- Training with next-character prediction and held-out validation.
- Checkpoints for saving, loading, and continuing training.
- Text generation with temperature and top-k sampling.
- CPU, Apple Silicon MPS, and NVIDIA CUDA device selection.

No pretrained weights are loaded when starting a new training run. The included demo checkpoint was trained with this project’s own code.

## How it learns

During training, each input character is paired with the character immediately after it:

```text
Text:     hello
Input:    h e l l
Target:   e l l o
```

The model turns character IDs into vectors, combines information from earlier positions through attention, and produces scores for the next character. A causal mask prevents it from reading future characters. Cross-entropy loss measures prediction error, and the optimizer adjusts the weights to reduce that error.

```text
Text → character IDs → character + position embeddings
     → Transformer blocks → next-character scores → sampled character
```

Generation repeats the last step: append a sampled character, feed the updated text back into the model, and continue.

## Build your first model

The commands below use Python 3.12 and a macOS/Linux shell. Run them from the project folder after cloning.

### 1. Set up the project

```bash
git clone https://github.com/Yoken0/kenAI.git
cd kenAI
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. Get the training text

Start with [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), a public practice corpus of approximately 1.1 million characters.

```bash
.venv/bin/python -m kenai.download
```

The repository already includes this corpus at `data/shakespeare.txt`; the command keeps an existing file. Its source and checksum are recorded in [data/shakespeare.source.json](data/shakespeare.source.json). Dependencies and missing datasets require an internet connection; training and generation run locally.

### 3. Train from random weights

```bash
.venv/bin/python -m kenai.train \
  --data data/shakespeare.txt \
  --out runs/first-model \
  --steps 2000
```

This creates a **new model** in `runs/first-model`. It does not load the bundled checkpoint. Each step updates the weights using a batch of text. Expect rough output at first; 2,000 steps is a starting experiment, not a guarantee of fluent writing.

The trainer uses the first 90% of the corpus for training and the final 10% for validation. These are separate, contiguous text segments. Watch validation loss to see whether predictions on held-out text improve. Because validation is used to select checkpoints, it is not a separate final test set.

The run folder contains:

| File | Purpose |
| --- | --- |
| `best.pt` | Model with the lowest measured validation loss; use it to generate text. |
| `latest.pt` | Most recently saved model and optimizer state; use it to resume training. |
| `metrics.jsonl` | Training and validation loss measurements. |

Existing run folders are protected from overwriting. To start another experiment, choose a new `--out` folder. To continue this one, use `--resume`. Pressing `Ctrl+C` during training saves a checkpoint before stopping.

### 4. Generate text

```bash
.venv/bin/python -m kenai.generate \
  --checkpoint runs/first-model/best.pt \
  --prompt "ROMEO:" \
  --tokens 400 \
  --temperature 0.8
```

One token is one character in this implementation. `--tokens 400` asks for 400 new characters after the prompt. Use prompt characters that exist in the training vocabulary.

Lower temperatures, such as `0.6`, favor more probable characters. Higher temperatures, such as `1.0`, produce more varied output. `--top-k 20` limits each sampling step to the 20 highest-scoring candidates. Compare several prompts and settings alongside validation loss to understand what the model has learned.

### 5. Keep training

```bash
.venv/bin/python -m kenai.train \
  --resume runs/first-model/latest.pt \
  --steps 5000
```

`--steps` is the **total target**. A checkpoint at step 2,000 needs 3,000 more updates to reach 5,000. Resume with the same corpus and architecture. To change the dataset or model size, start a new run.

## Try the included model

A trained example is included in `runs/shakespeare/` so you can try generation before running your own experiment:

```bash
.venv/bin/python -m kenai.generate \
  --checkpoint runs/shakespeare/best.pt \
  --prompt "ROMEO:" \
  --tokens 400
```

This example has **826,368 parameters** and completed **2,000 training steps** on Apple MPS. Its validation loss fell from **4.2400 to 1.9931**. The output is still rough, with fragments of words and Shakespeare-like formatting rather than consistently coherent prose.

See the [generated sample](runs/shakespeare/sample.txt) and [training measurements](runs/shakespeare/metrics.jsonl). The example demonstrates the training pipeline; it has not learned anyone’s personal writing style.

## Use your own writing

To experiment with a personal style, collect examples as UTF-8 `.txt` or `.md` files. Choose text representative of the writing you want to generate, and remove unwanted boilerplate or pasted material.

```bash
.venv/bin/python -m kenai.prepare "/path/to/my-writing" \
  --output data/my_style.txt

.venv/bin/python -m kenai.train \
  --data data/my_style.txt \
  --out runs/my-style \
  --steps 2000

.venv/bin/python -m kenai.generate \
  --checkpoint runs/my-style/best.pt \
  --prompt "I " \
  --tokens 400
```

Preparation combines the documents and skips empty or exactly duplicated text. Review the combined corpus before training. More varied, relevant writing provides more examples to learn from; very small datasets can lead to memorization. See [the data guide](data/README.md) for preparation details.

## Model sizes and hardware

| Preset | Parameters with the Shakespeare vocabulary | Context in characters | Embedding width | Attention heads | Transformer blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `starter` | 826,368 | 128 | 128 | 4 | 4 |
| `studio` | 4,837,888 | 256 | 256 | 8 | 6 |

`starter` is the default and was trained on an Apple Silicon Mac with 16 GB of memory. Both presets default to a batch size of 16. The larger `studio` preset is an optional next experiment for a machine such as an M2 Mac with 32 GB of memory:

```bash
.venv/bin/python -m kenai.train \
  --data data/shakespeare.txt \
  --out runs/studio-model \
  --preset studio \
  --steps 2000
```

The larger preset has passed a full-context forward/backward check on CPU; a full training run on the planned 32 GB machine has not been verified. Changing presets creates a new model. More memory allows larger experiments, while generation quality still depends on the data and training.

`--device auto` selects CUDA when available, then Apple MPS, otherwise CPU. Use `--device cpu` to force CPU execution. Reduce `--batch-size`, for example to `4`, if memory is tight. `--threads 4` controls CPU threads and is the default.

Explore the remaining settings with:

```bash
.venv/bin/python -m kenai.train --help
.venv/bin/python -m kenai.generate --help
```

## Continue on another machine

Clone the repository and recreate `.venv` using the setup commands. The public Shakespeare corpus and example checkpoints are included in Git, so the bundled model is ready to load after installing dependencies.

For your own experiments, also copy the relevant corpus and run folder. Other datasets and run folders are ignored by Git by default. Run commands from the new project folder, or use `--data` to point to the moved corpus when resuming.

Checkpoints store model weights, optimizer state, vocabulary, and step count. Tensors are saved on CPU so they can be loaded onto another supported device. A copied GPU checkpoint has been verified to resume on CPU in a different folder. Exact numerical reproduction across devices is not guaranteed; accelerator random-number state is not currently saved.

## Explore the implementation

| File | What to study |
| --- | --- |
| [tokenizer.py](kenai/tokenizer.py) | Building a vocabulary and converting between text and character IDs. |
| [model.py](kenai/model.py) | Embeddings, attention, Transformer blocks, prediction loss, and sampling. |
| [train.py](kenai/train.py) | Batching, optimization, validation, and checkpoints. |
| [generate.py](kenai/generate.py) | Loading a trained model and continuing a prompt. |
| [prepare.py](kenai/prepare.py) | Combining your writing into a training corpus. |

The tests check causal attention, gradient flow, learning a small pattern, tokenizer round trips, checkpoint loading, CPU resumption, and corpus integrity.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Inspiration and learning resources

- [Green Code — I Built an LLM from Scratch](https://www.youtube.com/watch?v=s9w3gtgvNSU): the inspiration for this project.
- [Andrej Karpathy’s char-rnn](https://github.com/karpathy/char-rnn): source of the Tiny Shakespeare practice corpus.
- [Sebastian Raschka — Build a Large Language Model (From Scratch)](https://sebastianraschka.com/llms-from-scratch/).
- [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/).
- [Andrej Karpathy’s videos](https://www.youtube.com/@AndrejKarpathy).
