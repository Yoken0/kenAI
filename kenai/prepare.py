"""Combine your UTF-8 writing into a local training corpus."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(source: Path, output: Path) -> dict:
    if not source.exists():
        raise ValueError(f"Writing not found: {source}")
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
        and not any(part.startswith(".") for part in p.relative_to(source).parts)
    )
    output = output.resolve()
    files = [p for p in files if p.resolve() != output]
    documents, seen = [], set()
    for path in files:
        # Read strictly: silently dropping undecodable text can ruin a corpus.
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text and digest not in seen:
            documents.append(text)
            seen.add(digest)
    if not documents:
        raise ValueError("No nonempty UTF-8 .txt or .md writing found.")
    corpus = "\n\n".join(documents) + "\n"
    if len(corpus) < 1500:
        raise ValueError("Use at least 1,500 characters for a test; much more writing is better.")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    metadata = {
        "documents": len(documents), "characters": len(corpus),
        "unique_characters": len(set(corpus)),
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A .txt/.md file or a folder of writing")
    parser.add_argument("--output", type=Path, default=Path("data/my_style.txt"))
    args = parser.parse_args()
    try:
        metadata = prepare(args.source, args.output)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Preparation error: {exc}\n")
    print(f"Prepared {metadata['documents']} documents, {metadata['characters']:,} characters → {args.output}")
    if metadata["characters"] < 100_000:
        print("This is a small corpus: expect rough output and possible memorization. Add more writing when you can.")


if __name__ == "__main__":
    main()
