from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert resources manifest into Qwen3-TTS train_raw.jsonl format")
    parser.add_argument("--manifest", required=True, help="Input manifest jsonl")
    parser.add_argument("--clips_dir", required=True, help="Directory containing segmented wav clips")
    parser.add_argument("--output_jsonl", default=None, help="Output train_raw jsonl path")
    parser.add_argument("--shared_ref_audio", default=None, help="Shared reference audio for all rows (recommended)")
    parser.add_argument("--fallback_ref_mode", choices=["error", "same_as_audio", "first_clip"], default="error")
    parser.add_argument("--text_field", default="seed_text", help="Text field in source manifest")
    parser.add_argument("--audio_field", default="audio_path", help="Audio field in source manifest")
    parser.add_argument("--source_ref_field", default="source_audio", help="Reference audio field in source manifest")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_audio_path(raw_path: str, clips_dir: Path) -> Path:
    candidate = Path(raw_path)
    remapped = clips_dir / candidate.name
    if remapped.exists():
        return remapped
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Clip not found for source path: {raw_path}")


def resolve_ref_audio(
    row: dict[str, Any],
    args: argparse.Namespace,
    clips_dir: Path,
    first_clip: Path | None,
    resolved_audio: Path,
) -> Path:
    if args.shared_ref_audio:
        shared_ref = Path(args.shared_ref_audio)
        if not shared_ref.exists():
            raise FileNotFoundError(f"Shared ref audio not found: {shared_ref}")
        return shared_ref

    source_ref = row.get(args.source_ref_field)
    if source_ref:
        source_ref_path = Path(source_ref)
        if source_ref_path.exists():
            return source_ref_path
        remapped = clips_dir.parent / source_ref_path.name
        if remapped.exists():
            return remapped

    if args.fallback_ref_mode == "same_as_audio":
        return resolved_audio
    if args.fallback_ref_mode == "first_clip":
        if first_clip is None:
            raise FileNotFoundError("Unable to resolve first clip for fallback ref mode")
        return first_clip

    raise FileNotFoundError(
        "Reference audio is missing. Please provide --shared_ref_audio or use a fallback_ref_mode."
    )


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    clips_dir = Path(args.clips_dir)
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")

    first_clip = None
    converted: list[dict[str, Any]] = []
    missing_ref = 0
    stale_audio_paths = 0

    for idx, row in enumerate(rows):
        resolved_audio = resolve_audio_path(str(row[args.audio_field]), clips_dir)
        if Path(row[args.audio_field]) != resolved_audio:
            stale_audio_paths += 1
        if first_clip is None:
            first_clip = resolved_audio

        try:
            ref_audio = resolve_ref_audio(row, args, clips_dir, first_clip, resolved_audio)
        except FileNotFoundError:
            missing_ref += 1
            if args.dry_run:
                ref_audio = Path("<missing>")
            else:
                raise

        text = str(row.get(args.text_field, "")).strip()
        if not text:
            raise ValueError(f"Missing text for row {idx}: {row}")

        converted.append(
            {
                "audio": str(resolved_audio),
                "text": text,
                "ref_audio": str(ref_audio),
                "language": "Japanese",
                "utt": row.get("utt", f"row_{idx:05d}"),
                "duration": row.get("duration"),
            }
        )

    print(f"rows={len(converted)}")
    print(f"stale_audio_paths={stale_audio_paths}")
    print(f"missing_ref_rows={missing_ref}")
    print(f"first_audio={converted[0]['audio']}")
    print(f"first_text={converted[0]['text']}")
    print(f"first_ref_audio={converted[0]['ref_audio']}")

    if not args.dry_run:
        if not args.output_jsonl:
            raise ValueError("--output_jsonl is required unless --dry_run is set")
        dump_jsonl(converted, Path(args.output_jsonl))
        print(f"Saved converted jsonl to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
