from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with .txt files")
    ap.add_argument("--output_file", required=True, help="Output JSONL path")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.txt"))

    with open(args.output_file, "w", encoding="utf-8") as out:
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            text = text.strip()
            if not text:
                continue
            rec = {"text": text}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()