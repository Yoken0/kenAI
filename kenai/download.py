"""Download the public Tiny Shakespeare practice corpus (about 1 MB)."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/shakespeare.txt"))
    args = parser.parse_args()
    if args.output.exists():
        print(f"Already present: {args.output}. Keeping the existing file.")
        return
    try:
        with urlopen(SOURCE, timeout=60) as response:
            content = response.read(5_000_001)
        if not 100_000 < len(content) < 5_000_000:
            raise ValueError("Unexpected dataset size; no file was written.")
        text = content.decode("utf-8")
        if not text.startswith("First Citizen:"):
            raise ValueError("Unexpected dataset contents; no file was written.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        metadata = {"source": SOURCE, "characters": len(text), "sha256": hashlib.sha256(content).hexdigest()}
        args.output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Download error: {exc}\n")
    print(f"Downloaded {len(text):,} characters → {args.output}")
    print("This is Shakespeare practice data. It does not represent your personal writing style.")


if __name__ == "__main__":
    main()
