Place your writing here as UTF-8 `.txt` or `.md` files, or prepare it from another folder:

```bash
.venv/bin/python -m kenai.prepare "/path/to/my-writing" --output data/my_style.txt
```

The input is copied, never edited or uploaded. Exact duplicate documents and empty files are skipped. Markdown is kept as written, so remove front matter, navigation, and pasted text if you do not want the model to learn them. Review the resulting corpus before training.

Use writing you want to imitate. Start with hundreds of thousands of characters if available; a tiny corpus is useful for learning the mechanics but usually leads to memorization and poor generation. This is a practical starting suggestion, not a quality guarantee.

The public `shakespeare.txt` practice corpus and its `shakespeare.source.json` attribution are included in Git. Other files in this directory are ignored by default, so future personal writing stays local unless you deliberately add it. No personal writing is included in the starter project.
