from __future__ import annotations

import argparse
import os
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with .txt files")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--model_type", default="unigram", choices=["unigram", "bpe"]) 
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_glob = str(Path(args.input_dir) / "*.txt")
    out_prefix = str(Path(args.output_dir) / "spm")

    spm.SentencePieceTrainer.Train(
        input=input_glob,
        model_prefix=out_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=0.9995,
        byte_fallback=True,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

    print(f"Tokenizer saved to {out_prefix}.model and {out_prefix}.vocab")


if __name__ == "__main__":
    main()