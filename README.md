# kenAI — build and train a small language model

This project builds a small GPT from scratch: its weights start randomly, it learns to predict the next character, and it generates text using the weights you train. It uses PyTorch for tensor operations and automatic differentiation. There are no pretrained model weights or hosted AI APIs.

It follows the from-scratch direction of Green Code’s [“I Built an LLM from Scratch”](https://www.youtube.com/watch?v=s9w3gtgvNSU). It is a deliberately small first implementation, not a verified reproduction of the video’s model.

You do not need your own dataset to begin. Start with the public Tiny Shakespeare practice corpus below. That model will learn patterns from Shakespeare; learning **your own writing style** requires examples of your writing later. This is an educational text-completion model, not a general-purpose chat assistant.

If `runs/shakespeare/best.pt` already exists in this copy, the initial setup and practice training have been done. Jump to section 3 to generate text. To train longer, set `--steps` above the saved step; to start over, choose a new `--out` folder. Existing runs are protected from accidental overwriting.

The initial local run completed **2,000 steps** on Apple MPS with **826,368 parameters**. Validation loss improved from **4.2400 to 1.9931**. All **13 tests passed**, including saving, reloading, and resuming on CPU. The 4,837,888-parameter `studio` preset also passed a full-context forward/backward check on CPU; training it on the future 32 GB machine remains to be tried. These checks confirm the workflow works, not that the generated text is fluent. See `runs/shakespeare/sample.txt` for a sample and `runs/shakespeare/metrics.jsonl` for measurements.

## 1. Set up Python

Run these commands in the `kenAI` project folder. You need Python 3.12 installed; `python3.12 --version` checks that it is available.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Download practice text and train

```bash
.venv/bin/python -m kenai.download
.venv/bin/python -m kenai.train --data data/shakespeare.txt --out runs/shakespeare --steps 500
```

The download command saves [Tiny Shakespeare from Karpathy’s char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) to `data/shakespeare.txt`. Downloading the corpus and installing dependencies require internet access; training and generation run locally.

Start with the default `starter` preset on your current 16 GB machine. The first 500 steps are a practice run, not a promise of fluent output. A *step* is one update to the model after it predicts a batch of text.

The trainer saves these files in `runs/shakespeare`:

- `latest.pt`: the most recently saved checkpoint, used to resume training.
- `best.pt`: the checkpoint with the lowest measured validation loss.
- `metrics.jsonl`: training and validation measurements, including initial and final evaluations.

The final 10% of the corpus is a contiguous validation split; the earlier 90% is training text. Validation loss measures prediction error on held-out text, and lower is better. Because you use that split to choose checkpoints and settings, it is a development validation set. There is no separate final test set in this starter workflow.

## 3. Generate text, then train longer

```bash
.venv/bin/python -m kenai.generate --checkpoint runs/shakespeare/best.pt --prompt 'ROMEO:' --tokens 400
```

Here, one token is one character, so `--tokens 400` generates 400 new characters. A prompt is text to continue. It must use characters present in the training corpus. Expect rough or repetitive text early in training.

To continue the same run:

```bash
.venv/bin/python -m kenai.train --data data/shakespeare.txt --out runs/shakespeare --resume runs/shakespeare/latest.pt --steps 2000
```

`--steps` is the **total target**, not the number of additional steps. Resuming a checkpoint at step 500 with `--steps 2000` requests 1,500 more steps. If it is already at 2,000 steps, use a larger target such as `--steps 5000`. Keep the same corpus when resuming. Run commands from the project folder, or pass the correct corpus location with `--data`.

## 4. Train on your own writing later

Collect writing you want the model to imitate as UTF-8 `.txt` or `.md` files. Remove unwanted navigation, boilerplate, and pasted material. More varied, relevant writing gives the model more to learn; very small collections can be memorized.

```bash
.venv/bin/python -m kenai.prepare "/path/to/my-writing" --output data/my_style.txt
.venv/bin/python -m kenai.train --data data/my_style.txt --out runs/my-style --steps 2000
.venv/bin/python -m kenai.generate --checkpoint runs/my-style/best.pt --prompt 'I ' --tokens 400
```

Preparation combines the files and skips empty or exactly duplicated documents. Review `data/my_style.txt` before training. Use a new run for this corpus so you build its vocabulary and model from scratch. Without your own examples, the practice model cannot learn your personal style.

## Hardware and settings

`--device auto` selects an available accelerator (Apple MPS or NVIDIA CUDA), with CPU as the fallback. To choose CPU explicitly, add `--device cpu` to a training or generation command. `--threads 4` controls CPU threads and is the default. If memory is tight, try a smaller batch, such as `--batch-size 4`.

| Preset | Context in characters | Embedding width | Attention heads | Transformer blocks | Default batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `starter` (default) | 128 | 128 | 4 | 4 | 16 |
| `studio` | 256 | 256 | 8 | 6 | 16 |

After the smaller model works, you can try the optional `studio` preset on the planned M2 machine with 32 GB of memory:

```bash
.venv/bin/python -m kenai.train --data data/shakespeare.txt --out runs/shakespeare-studio --preset studio --steps 2000
```

This starts a new, larger model. A preset change does not enlarge an existing checkpoint. Even with 32 GB, this remains an educational model; building a capable general-purpose assistant requires substantially more data, compute, and training work.

See all options with `.venv/bin/python -m kenai.train --help`. Training also accepts `--block-size`, `--learning-rate`, `--eval-every`, `--eval-batches`, and `--seed`.

## Move the project to your next Mac

Copy the project, including `data/` and `runs/` if you want to resume your training. Exclude `.venv/` and `__pycache__/` folders, then repeat the Python setup commands on the destination machine. Checkpoints save weights on CPU so they can be loaded onto another supported device. Run the resume command from the copied project folder using the same corpus and a larger total step target.

Weights, optimizer state, vocabulary, and step count survive the move. Exact numerical reproduction across devices is not guaranteed; accelerator random-number state is not currently saved.

The public Shakespeare corpus and the trained `runs/shakespeare/` checkpoint are included in Git. You can also transfer this starter project by cloning the repository on your next Mac:

```bash
git clone https://github.com/Yoken0/kenAI.git
cd kenAI
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m kenai.generate --prompt 'ROMEO:' --tokens 400
```

Other datasets and run folders are ignored by default. Copy those separately if you create personal training data or additional models later. The local Python environment and caches are excluded from Git and recreated by the setup commands.

## Learn how it works

Read the code in this order:

1. `kenai/tokenizer.py`: map each distinct character to a number and back.
2. `kenai/model.py`: turn character and position embeddings into predictions through causal attention and feed-forward blocks, with residual connections and normalization. Causal attention prevents a position from reading future characters.
3. `kenai/train.py`: predict the next character, measure cross-entropy loss, update weights, and evaluate on held-out text.
4. `kenai/generate.py`: load your checkpoint and repeatedly sample the next character to extend a prompt.

Run the project’s tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

For deeper study, the video’s description recommends [Stanford CS336](https://cs336.stanford.edu/), [Sebastian Raschka’s LLMs from Scratch](https://sebastianraschka.com/llms-from-scratch/), and [Andrej Karpathy’s videos](https://www.youtube.com/@AndrejKarpathy).
