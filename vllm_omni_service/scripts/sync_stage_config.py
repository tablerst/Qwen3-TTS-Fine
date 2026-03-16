from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the official vllm-omni qwen3_tts stage config into this repository."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional explicit source path to qwen3_tts.yaml. If omitted, the script locates the installed vllm_omni package.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qwen3_tts.stage.yaml"),
        help="Output path for the copied stage config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def find_installed_stage_config(explicit_source: str | None) -> Path:
    if explicit_source:
        source = Path(explicit_source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Explicit source path does not exist: {source}")
        return source

    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate installed package 'vllm_omni'. Install vllm-omni first, or pass --source explicitly."
        )

    package_root = Path(spec.origin).resolve().parent
    stage_config_path = package_root / "model_executor" / "stage_configs" / "qwen3_tts.yaml"
    if not stage_config_path.exists():
        raise FileNotFoundError(
            f"Installed vllm_omni package was found at {package_root}, but qwen3_tts.yaml was not found under model_executor/stage_configs/."
        )
    return stage_config_path


def main() -> None:
    args = parse_args()
    source = find_installed_stage_config(args.source)
    output = Path(args.output).expanduser().resolve()

    if output.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {output}. Use --force to overwrite it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    print(f"Copied stage config from: {source}")
    print(f"Saved local stage config to: {output}")
    print("You can now point vllm-omni --stage-configs-path to this local file.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
